"""Pure data + state helpers for the Correlations tab.

CHARTS: three histogram charts (slots y1/y2/y3), each with a picked Y
measure, sharing one X mode:
  x = 'token_count'  -> each chart bins requests over ITS OWN y measure's
                        values; bar height = request count.
  x = positional     -> all charts share bins over turn # / cumulative time /
                        busy time; bar height = SUM of the chart's y measure
                        in the bin (how much happened when).
Measures are getters over the ENRICHED per-request rows produced by
deepdive_data.aggregate_selection (seq / start_s / busy_s / kv_bytes / flops),
so KV bytes and FLOPs follow the global serving assumptions.

SELECTIONS: color-coded partitions of ONE shared selection chart. The first
bin picked anchors store['chart'] to that slot; after that, bins are
EXCLUSIVE — assigning a bin to the armed inspector steals it from whichever
inspector held it (click a bin the armed inspector already owns to release
it). When every inspector is empty the anchor clears and the next click may
land on any chart. Conditioning always applies: every non-empty selection's
matching requests stack on the OTHER charts in its color. Inspectors never
disappear: at least one exists, clearing just empties it.

Store shape:
  {"live": sid, "next_sid": int, "chart": "y1"|"y2"|"y3"|None,
   "selections": [{"sid": int, "bins": [int, ...]}]}

Every mutation returns a NEW store (copy-on-write). Dash-free module — this
is the unit-test surface.
"""
from __future__ import annotations

from modules.binning import bin_index
from modules.theme import PALETTE

MAX_SELECTIONS = 10
CHART_SLOTS = ("y1", "y2", "y3")

# --- Measure registry (getters over enriched per-request rows) --------------

X_MEASURES = {
    "token_count": dict(
        label="token count (per-measure bins)",
        info="Classic histogram mode: each chart bins requests over its OWN "
             "y measure's values (tokens, bytes, or FLOPs) and counts them. "
             "The three charts have independent bin edges.",
        getter=None),  # sentinel: bins come from each chart's own y measure
    "turn_number": dict(
        label="turn # (request seq in conv)",
        info="Bins requests by their sequence number within their "
             "conversation (main and subagent requests both count). All "
             "three charts share these bins, and bar height becomes the SUM "
             "of each chart's measure in the bin.",
        getter=lambda p: p["seq"]),
    "cumulative_time": dict(
        label="cumulative time (s)",
        info="Bins requests by wall-clock seconds since their conversation's "
             "first request — idle stretches included. All three charts "
             "share these bins, and bar height becomes the SUM of each "
             "chart's measure in the bin.",
        getter=lambda p: p["start_s"]),
    "busy_time": dict(
        label="busy time (s, active only)",
        info="Bins requests by ACTIVE seconds elapsed at their start (time "
             "when at least one request of the conversation was in flight — "
             "idle gaps compressed out). All three charts share these bins, "
             "and bar height becomes the SUM of each chart's measure in the "
             "bin.",
        getter=lambda p: p["busy_s"]),
}

Y_MEASURES = {
    "kv_cache_bytes": dict(
        label="KV cache size (bytes)",
        info="KV-cache bytes of the request's context under the selected "
             "architecture and KV precision (from the Explorer assumption "
             "bar).",
        getter=lambda p: p["kv_bytes"]),
    "kv_cache_tokens": dict(
        label="KV cache (tokens)",
        info="Context tokens held in the KV cache — the request's total "
             "input (cached + new). Architecture-independent; the byte size "
             "is this times KV bytes/token under the assumptions.",
        getter=lambda p: p["in_tokens"]),
    "new_input": dict(
        label="uncached input (tokens)",
        info="Uncached input tokens — new text (user message, tool results, "
             "agent hand-offs) not already served from the prompt cache; "
             "this is what prefill actually computes.",
        getter=lambda p: p["uncached_tokens"]),
    "decode_output": dict(
        label="decode output (tokens)",
        info="Tokens decoded (generated) by the request.",
        getter=lambda p: p["out_tokens"]),
    "turn_flops": dict(
        label="turn FLOPs",
        info="Implied FLOPs of the request (prefill + decode) under the "
             "selected architecture from the Explorer assumption bar.",
        getter=lambda p: p["flops"]),
}

ALL_MEASURES = {**X_MEASURES, **Y_MEASURES}
DEFAULT_AXES = {"x": "turn_number", "y1": "kv_cache_tokens",
                "y2": "new_input", "y3": "decode_output"}


def measure_values(rows: list[dict], key: str) -> list[float]:
    """One measure over enriched rows. KeyError on unknown key; ValueError on
    the x-sentinel (it has no getter — each chart bins its own y measure)."""
    getter = ALL_MEASURES[key]["getter"]
    if getter is None:
        raise ValueError(f"measure {key!r} is a binning mode, not a value")
    return [getter(p) for p in rows]


# --- Selection store ---------------------------------------------------------

def selection_color(sid: int) -> str:
    """Stable color for a selection id (survives clearing other selections)."""
    return PALETTE[sid % len(PALETTE)]


def _empty_selection(sid: int) -> dict:
    return {"sid": sid, "bins": []}


def initial_store() -> dict:
    """One empty live selection, no anchored chart."""
    return {"live": 0, "next_sid": 1, "chart": None,
            "selections": [_empty_selection(0)]}


def _copy(store: dict) -> dict:
    return {"live": store["live"], "next_sid": store["next_sid"],
            "chart": store["chart"],
            "selections": [dict(s, bins=list(s["bins"]))
                           for s in store["selections"]]}


def get_selection(store: dict, sid: int) -> dict:
    for s in store["selections"]:
        if s["sid"] == sid:
            return s
    raise KeyError(f"no selection with sid={sid}")


def _unanchor_if_all_empty(store: dict) -> None:
    if all(not s["bins"] for s in store["selections"]):
        store["chart"] = None


def _anchor(store: dict, chart: str) -> None:
    if chart not in CHART_SLOTS:
        raise KeyError(f"unknown chart slot {chart!r}")
    if store["chart"] is None:
        store["chart"] = chart
    if store["chart"] != chart:
        raise ValueError(
            f"selections live in chart {store['chart']} — this click "
            f"targeted {chart}")


def assign_bin(store: dict, sid: int, chart: str, bin_idx: int) -> dict:
    """The armed inspector claims a bin: any prior owner loses it (bins are
    exclusive); claiming a bin the inspector already owns releases it. First
    bin anchors the shared selection chart; a click on another chart raises
    ValueError (the caller renders those charts click-inert)."""
    out = _copy(store)
    _anchor(out, chart)
    owner = get_selection(out, sid)
    bin_idx = int(bin_idx)
    already_owned = bin_idx in owner["bins"]
    for s in out["selections"]:
        if bin_idx in s["bins"]:
            s["bins"].remove(bin_idx)
    if not already_owned:
        owner["bins"] = sorted(owner["bins"] + [bin_idx])
    _unanchor_if_all_empty(out)
    return out


def assign_bin_range(store: dict, sid: int, chart: str,
                     lo_bin: int, hi_bin: int) -> dict:
    """Box-select: the armed inspector claims the inclusive bin range
    (stealing from other inspectors; no release semantics for ranges)."""
    if hi_bin < lo_bin:
        lo_bin, hi_bin = hi_bin, lo_bin
    out = _copy(store)
    _anchor(out, chart)
    claimed = set(range(int(lo_bin), int(hi_bin) + 1))
    for s in out["selections"]:
        if s["sid"] != sid:
            s["bins"] = [b for b in s["bins"] if b not in claimed]
    owner = get_selection(out, sid)
    owner["bins"] = sorted(set(owner["bins"]) | claimed)
    return out


def add_selection(store: dict) -> dict:
    """Append a new empty selection and arm it."""
    out = _copy(store)
    if len(out["selections"]) >= MAX_SELECTIONS:
        raise ValueError(f"at most {MAX_SELECTIONS} selections")
    sid = out["next_sid"]
    out["next_sid"] += 1
    out["selections"].append(_empty_selection(sid))
    out["live"] = sid
    return out


def clear_selection(store: dict, sid: int) -> dict:
    """Empty a selection. The section stays; the shared chart anchor clears
    when every selection is empty."""
    out = _copy(store)
    get_selection(out, sid)["bins"] = []
    _unanchor_if_all_empty(out)
    return out


def set_live(store: dict, sid: int) -> dict:
    """Arm an inspector: its colored cursor does the next bin picks."""
    out = _copy(store)
    get_selection(out, sid)  # KeyError on phantom sid
    out["live"] = sid
    return out


# --- Membership --------------------------------------------------------------

def member_mask(values: list[float], edges: list[float], bins: list[int],
                log_x: bool) -> list[bool]:
    """Which rows fall in the selected bins of the selection chart's binning
    (bin_index rules, so it matches what the chart drew). Empty bins or
    out-of-range indices raise — matching against nothing is a caller bug."""
    if not bins:
        raise ValueError("member_mask with no bins selected")
    n = len(edges) - 1
    bad = [b for b in bins if not 0 <= b < n]
    if bad:
        raise ValueError(f"bin indices out of range 0..{n - 1}: {bad}")
    bset = set(bins)
    return [bin_index(edges, v, log_x) in bset for v in values]


def bin_runs(bins: list[int]) -> list[tuple[int, int]]:
    """Bin indices -> contiguous inclusive (first, last) runs, ascending.
    [0, 1, 2, 5, 7, 8] -> [(0, 2), (5, 5), (7, 8)]."""
    runs: list[tuple[int, int]] = []
    for b in sorted(set(bins)):
        if runs and b == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((b, b))
    return runs
