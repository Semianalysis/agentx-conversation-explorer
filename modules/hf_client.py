"""Internal-mode HuggingFace source: list org datasets (public + whatever the
user's own HF token can read) and import raw cc-traces-weka datasets into the
SAME local cache layout the AgentX API importer uses — downstream code cannot
tell them apart.

GATING (the public app ships with all of this OFF):
- `AGENTX_INTERNAL=1` (env) reveals the Overview "Internal sources" panel.
  No unpublished dataset name exists anywhere in this codebase — listings are
  fetched live from HF at click time.
- `HF_TOKEN` (env, standard HF variable) is the actual access key: private /
  unreleased datasets appear only if the USER'S token can read them, so the
  security boundary is HuggingFace's own ACL, per user, never app code.
- `AGENTX_HF_SOURCES` (env, comma-separated authors/namespaces, default
  `semianalysisai`) chooses whose datasets to list — point it at your own
  namespace to explore traces of your own work.

Imported datasets get slug `hf--<name>` under data/, with a locally computed
detail.json (summary stats), conversations_index.json (token-sorted), and
conversations/*.json built by modules/weka_raw (the port of the site's own
ingest). Truncated API cells raise — never fabricate.
"""
from __future__ import annotations

import json
import logging
import os

import requests

from modules.api_client import DATA_DIR
from modules.weka_raw import build_conversation_structure

logger = logging.getLogger(__name__)

_ROWS_API = "https://datasets-server.huggingface.co/rows"
_HF_API = "https://huggingface.co/api"
_TIMEOUT_S = 120
_PAGE_ROWS = 5  # raw rows are MBs each — small pages avoid cell truncation


def internal_mode() -> bool:
    return os.environ.get("AGENTX_INTERNAL", "").strip() not in ("", "0", "false")


def hf_token() -> str | None:
    return (os.environ.get("HF_TOKEN") or
            os.environ.get("HUGGINGFACE_TOKEN") or None)


def hf_sources() -> list[str]:
    raw = os.environ.get("AGENTX_HF_SOURCES", "semianalysisai")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _headers() -> dict:
    tok = hf_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def hf_slug(dataset_id: str) -> str:
    return "hf--" + dataset_id.rsplit("/", 1)[-1]


def is_weka_family(dataset_id: str) -> bool:
    """Only the cc-traces-weka raw schema is supported for import (the other
    org datasets — swebench traces, WildChat token counts — have different
    schemas; they are LISTED but refuse import rather than mis-parse)."""
    return "cc-traces" in dataset_id.rsplit("/", 1)[-1]


def list_source_datasets() -> list[dict]:
    """Datasets of every configured source namespace, newest first. With an
    HF_TOKEN this includes private datasets the token can read."""
    out = []
    for author in hf_sources():
        r = requests.get(f"{_HF_API}/datasets",
                         params={"author": author, "full": "false",
                                 "limit": "500"},
                         headers=_headers(), timeout=_TIMEOUT_S)
        r.raise_for_status()
        for d in r.json():
            out.append({
                "id": d["id"],
                "slug": hf_slug(d["id"]),
                "private": bool(d.get("private")),
                "updated": (d.get("lastModified") or "")[:10],
                "importable": is_weka_family(d["id"]),
                "imported_convs": len(list(
                    (DATA_DIR / hf_slug(d["id"]) / "conversations").glob("*.json")))
                if (DATA_DIR / hf_slug(d["id"]) / "conversations").is_dir() else 0,
            })
    out.sort(key=lambda d: d["updated"], reverse=True)
    return out


def _dataset_license(dataset_id: str) -> str | None:
    r = requests.get(f"{_HF_API}/datasets/{dataset_id}", headers=_headers(),
                     timeout=_TIMEOUT_S)
    if r.status_code != 200:
        return None
    return ((r.json().get("cardData") or {}).get("license")) or None


def _fetch_rows(dataset_id: str, offset: int, length: int) -> dict:
    r = requests.get(_ROWS_API,
                     params={"dataset": dataset_id, "config": "default",
                             "split": "train", "offset": offset,
                             "length": length},
                     headers=_headers(), timeout=_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def local_hf_datasets() -> list[dict]:
    """Registry cards for locally imported HF datasets (detail.json scan)."""
    out = []
    for p in sorted(DATA_DIR.glob("hf--*/detail.json")):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def import_dataset(dataset_id: str, limit: int | None = None) -> dict:
    """Download a raw weka dataset from HF and write the standard local cache
    (detail.json / conversations_index.json / conversations/*.json). Returns
    the detail card. Slow — rows are MBs each. Raises on unsupported schema,
    truncated cells, or invariant violations; partial caches are overwritten
    on re-run (idempotent)."""
    if not is_weka_family(dataset_id):
        raise ValueError(f"{dataset_id}: not a cc-traces-weka-family dataset "
                         "— raw schema not supported")
    slug = hf_slug(dataset_id)
    conv_dir = DATA_DIR / slug / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)

    index_items, model_mix, req_counts = [], {}, []
    totals = {"in": 0, "out": 0, "cached": 0, "uncached": 0,
              "mainTurns": 0, "subagentTurns": 0, "subagentGroups": 0}
    block_size = None
    offset, n_rows = 0, None
    while n_rows is None or offset < n_rows:
        page = _fetch_rows(dataset_id, offset, min(_PAGE_ROWS,
                                                   (limit or 10**9) - offset))
        n_rows = page["num_rows_total"] if limit is None else min(
            limit, page["num_rows_total"])
        for item in page["rows"]:
            if item.get("truncated_cells"):
                raise ValueError(
                    f"{dataset_id} row {item['row_idx']}: cells truncated by "
                    f"the HF rows API ({item['truncated_cells']}) — refusing "
                    "to import partial data; lower _PAGE_ROWS")
            row = item["row"]
            structure = build_conversation_structure(row)
            t = structure["totals"]
            if t["in"] != t["cached"] + t["uncached"]:
                raise ValueError(f"{row['id']}: in != cached + uncached")
            block_size = structure["blockSize"]
            with open(conv_dir / f"{row['id']}.json", "w",
                      encoding="utf-8") as f:
                json.dump({"conv_id": row["id"], "structure": structure}, f)
            sub_turns = [c for n in structure["nodes"]
                         if n["kind"] == "subagent" for c in n["children"]]
            index_items.append({
                "conv_id": row["id"], "models": row.get("models") or [],
                "num_turns": t["numTurns"],
                "num_subagent_groups": t["numSubagentGroups"],
                "total_in": t["in"], "total_out": t["out"],
                "total_cached": t["cached"],
            })
            for n in structure["nodes"]:
                turns = n["children"] if n["kind"] == "subagent" else [n]
                for c in turns:
                    if c.get("model"):
                        model_mix[c["model"]] = model_mix.get(c["model"], 0) + 1
            req_counts.append(t["numTurns"] + len(sub_turns))
            for k_src, k_dst in (("in", "in"), ("out", "out"),
                                 ("cached", "cached"), ("uncached", "uncached")):
                totals[k_dst] += t[k_src]
            totals["mainTurns"] += t["numTurns"]
            totals["subagentTurns"] += len(sub_turns)
            totals["subagentGroups"] += t["numSubagentGroups"]
        offset += len(page["rows"])
        if not page["rows"]:
            raise ValueError(f"{dataset_id}: rows API returned an empty page "
                             f"at offset {offset} of {n_rows}")
        logger.info("%s: imported %d/%s conversations", dataset_id, offset,
                    n_rows)

    if not index_items:
        raise ValueError(f"{dataset_id}: zero conversations imported")
    index_items.sort(key=lambda it: -it["total_in"])
    rc = sorted(req_counts)
    detail = {
        "id": dataset_id, "slug": slug,
        "label": dataset_id.rsplit("/", 1)[-1] + " (HF)",
        "variant": "hf-import", "source": "hf",
        "description": f"Imported directly from HuggingFace ({dataset_id}) "
                       "in internal mode — not published on the AgentX site.",
        "hf_url": f"https://huggingface.co/datasets/{dataset_id}",
        "license": _dataset_license(dataset_id),
        "conversation_count": len(index_items),
        "summary": {
            "totalIn": totals["in"], "totalOut": totals["out"],
            "totalCached": totals["cached"],
            "cachedPct": (totals["cached"] / totals["in"]) if totals["in"] else None,
            "mainTurns": totals["mainTurns"],
            "subagentTurns": totals["subagentTurns"],
            "subagentGroups": totals["subagentGroups"],
            "modelMix": model_mix,
            "medianRequestsPerConversation": rc[len(rc) // 2],
            "meanRequestsPerConversation": sum(rc) / len(rc),
            "blockSize": block_size,
        },
    }
    with open(DATA_DIR / slug / "detail.json", "w", encoding="utf-8") as f:
        json.dump(detail, f)
    with open(DATA_DIR / slug / "conversations_index.json", "w",
              encoding="utf-8") as f:
        json.dump(index_items, f)
    return detail
