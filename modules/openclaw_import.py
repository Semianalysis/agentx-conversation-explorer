"""Import an OpenClaw trajectory export (steps.csv + friends, zipped or a
directory) into the SAME local cache the AgentX importers write, so every
tab explores it unchanged.

Mapping (one conversation per agent lane — `session_id`):
    step (one model call)        -> turn node
    cacheRead                    -> cached tokens
    cacheWrite + input           -> uncached tokens (what prefill computes)
    cacheRead + cacheWrite + input -> in tokens (the prompt)
    output                       -> out tokens
    ts_end - step_latency_s      -> startS (lane-relative), + latency -> endS
    kind == 'subagent'           -> the lane's turns hang under a subagent
                                    node (role/agent_id/depth follow)

The in == cached + uncached invariant holds by construction, so the records
loader accepts these files exactly like AgentX ones.

NOT imported, because this app has no concept of them: tool calls (names,
categories, durations in tools.csv), per-run cost, compaction counts. The
pauses between turns ARE preserved as timing, so the idle-before-turn
measure works; what happens inside a pause is not representable here.

CLI:  python -m modules.openclaw_import <export.zip|dir> [--slug NAME]

Pure stdlib (no pandas/dash) — safe to import anywhere and unit-testable.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path

from modules.api_client import DATA_DIR

logger = logging.getLogger(__name__)

SLUG_PREFIX = "openclaw--"
_REQUIRED = ("session_id", "model", "ts_end", "step_latency_s", "input",
             "output", "cacheRead", "cacheWrite")
_SUBAGENT_KEY = re.compile(r"agent:[^:]+:subagent:(?P<agent_id>[\w-]+)")


def _num(value, field: str, row_no: int) -> float:
    """Blank -> 0.0 (the exporter leaves unused usage columns empty);
    anything non-numeric is a malformed row and raises."""
    if value is None or str(value).strip() == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"steps.csv row {row_no}: {field}={value!r} "
                         "is not a number")


def _epoch_s(ts: str, row_no: int) -> float:
    """ISO-8601 (the export writes ...Z) -> epoch seconds."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        raise ValueError(f"steps.csv row {row_no}: ts_end={ts!r} is not ISO-8601")


def read_steps(path: Path) -> list[dict]:
    """steps.csv rows from a zip or a directory. Raises when the file is
    missing or lacks the columns this importer depends on."""
    if path.is_dir():
        f = path / "steps.csv"
        if not f.is_file():
            raise FileNotFoundError(f"no steps.csv in {path}")
        text = f.read_text(encoding="utf-8")
    else:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == "steps.csv"]
            if not names:
                raise FileNotFoundError(f"no steps.csv inside {path.name}")
            text = z.read(names[0]).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("steps.csv has no rows")
    missing = [c for c in _REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f"steps.csv missing columns {missing} — is this an "
                         f"OpenClaw trajectory export? (got {sorted(rows[0])[:8]}…)")
    return rows


def build_conversations(rows: list[dict]) -> tuple[dict[str, dict], int]:
    """Lane -> AgentX-shaped structure. Returns ({conv_id: structure},
    n_skipped). Rows without a usable timestamp are skipped and counted —
    never silently defaulted to 0."""
    lanes: dict[str, list[dict]] = {}
    skipped = 0
    for i, r in enumerate(rows, start=2):   # 2 = first data line in the file
        cid = (r.get("session_id") or "").strip()
        ts = (r.get("ts_end") or "").strip()
        if not cid or not ts:
            skipped += 1
            continue
        start = _epoch_s(ts, i) - _num(r.get("step_latency_s"), "step_latency_s", i)
        cached = int(_num(r.get("cacheRead"), "cacheRead", i))
        fresh = int(_num(r.get("cacheWrite"), "cacheWrite", i)
                    + _num(r.get("input"), "input", i))
        lanes.setdefault(cid, []).append({
            "kind": "turn",
            "model": (r.get("model") or "").strip() or "unknown",
            "in": cached + fresh, "cached": cached, "uncached": fresh,
            "out": int(_num(r.get("output"), "output", i)),
            "_start": start,
            "_end": start + _num(r.get("step_latency_s"), "step_latency_s", i),
            "_role": (r.get("kind") or "").strip(),
            "_key": (r.get("session_key") or "").strip(),
        })

    out: dict[str, dict] = {}
    for cid, turns in lanes.items():
        turns.sort(key=lambda t: t["_start"])
        t0 = turns[0]["_start"]
        nodes = []
        for idx, t in enumerate(turns):
            nodes.append({
                "kind": "turn", "turnIndex": idx, "model": t["model"],
                "in": t["in"], "cached": t["cached"], "uncached": t["uncached"],
                "out": t["out"],
                "startS": round(t["_start"] - t0, 3),
                "endS": round(t["_end"] - t0, 3),
            })
        is_sub = turns[0]["_role"] == "subagent"
        if is_sub:
            m = _SUBAGENT_KEY.search(turns[0]["_key"] or "")
            nodes = [{
                "kind": "subagent", "label": "Subagent",
                "agentId": m.group("agent_id") if m else cid,
                "startS": nodes[0]["startS"], "endS": nodes[-1]["endS"],
                "in": sum(n["in"] for n in nodes),
                "out": sum(n["out"] for n in nodes),
                "cached": sum(n["cached"] for n in nodes),
                "uncached": sum(n["uncached"] for n in nodes),
                "children": nodes,
            }]
        leaves = nodes[0]["children"] if is_sub else nodes
        out[cid] = {
            "blockSize": None, "nodes": nodes,
            "totals": {
                "in": sum(n["in"] for n in leaves),
                "out": sum(n["out"] for n in leaves),
                "cached": sum(n["cached"] for n in leaves),
                "uncached": sum(n["uncached"] for n in leaves),
                "numTurns": 0 if is_sub else len(leaves),
                "numSubagentGroups": 1 if is_sub else 0,
            },
        }
    return out, skipped


def import_openclaw(path: str | Path, slug: str | None = None,
                    data_dir: Path | None = None) -> dict:
    """Write data/<slug>/{detail,conversations_index}.json + conversations/*.
    Returns the detail card. Re-running replaces the dataset's files."""
    path = Path(path)
    data_dir = data_dir or DATA_DIR
    slug = slug or (SLUG_PREFIX + re.sub(r"[^\w.-]+", "-", path.stem).strip("-"))
    structures, skipped = build_conversations(read_steps(path))
    if not structures:
        raise ValueError(f"{path}: no usable rows (all {skipped} skipped)")

    conv_dir = data_dir / slug / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    for stale in conv_dir.glob("*.json"):
        stale.unlink()

    index, model_mix, req_counts = [], {}, []
    totals = dict.fromkeys(("in", "out", "cached", "uncached"), 0)
    main_turns = sub_turns = sub_groups = 0
    for cid, st in structures.items():
        with open(conv_dir / f"{cid}.json", "w", encoding="utf-8") as f:
            json.dump({"conv_id": cid, "structure": st}, f)
        leaves = (st["nodes"][0]["children"]
                  if st["nodes"][0]["kind"] == "subagent" else st["nodes"])
        for n in leaves:
            model_mix[n["model"]] = model_mix.get(n["model"], 0) + 1
        t = st["totals"]
        for k in totals:
            totals[k] += t[k]
        main_turns += t["numTurns"]
        sub_turns += len(leaves) if t["numSubagentGroups"] else 0
        sub_groups += t["numSubagentGroups"]
        req_counts.append(len(leaves))
        index.append({
            "conv_id": cid, "models": sorted({n["model"] for n in leaves}),
            "num_turns": t["numTurns"] or len(leaves),
            "num_subagent_groups": t["numSubagentGroups"],
            "total_in": t["in"], "total_out": t["out"],
            "total_cached": t["cached"],
        })
    index.sort(key=lambda it: -it["total_in"])
    rc = sorted(req_counts)
    detail = {
        "id": str(path.name), "slug": slug, "source": "openclaw",
        "label": f"{path.stem} (OpenClaw import)", "variant": "openclaw",
        "description": ("Imported from an OpenClaw trajectory export: one "
                        "conversation per agent lane, one turn per model "
                        "call. Tool calls and costs are not represented — "
                        "pauses keep their timing but not their contents."),
        "hf_url": None, "license": None,
        "conversation_count": len(index),
        "summary": {
            "totalIn": totals["in"], "totalOut": totals["out"],
            "totalCached": totals["cached"],
            "cachedPct": (totals["cached"] / totals["in"]) if totals["in"] else None,
            "mainTurns": main_turns, "subagentTurns": sub_turns,
            "subagentGroups": sub_groups, "modelMix": model_mix,
            "medianRequestsPerConversation": rc[len(rc) // 2],
            "meanRequestsPerConversation": sum(rc) / len(rc),
            "blockSize": "n/a",
        },
    }
    with open(data_dir / slug / "detail.json", "w", encoding="utf-8") as f:
        json.dump(detail, f)
    with open(data_dir / slug / "conversations_index.json", "w",
              encoding="utf-8") as f:
        json.dump(index, f)
    logger.info("%s: %d conversations, %d skipped rows", slug, len(index),
                skipped)
    detail["skipped_rows"] = skipped
    return detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export", help="OpenClaw export .zip or directory")
    ap.add_argument("--slug", help=f"dataset slug (default {SLUG_PREFIX}<name>)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    d = import_openclaw(args.export, args.slug)
    s = d["summary"]
    print(f"imported {d['slug']}: {d['conversation_count']} conversations, "
          f"{s['mainTurns'] + s['subagentTurns']:,} model calls, "
          f"{d['skipped_rows']} rows skipped")
    print("pick it in the dataset dropdown on Explorer or Correlations")


if __name__ == "__main__":
    main()
