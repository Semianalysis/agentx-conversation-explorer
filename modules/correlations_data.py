"""Pure data + state helpers for the Correlations tab.

CHARTS: three histogram charts (slots y1/y2/y3 - displayed as x1/x2/x3),
each binning requests over ITS OWN picked X measure; bar height is ALWAYS
the request count in the bin (log or linear y). Measures are getters over the ENRICHED per-request rows produced by
deepdive_data.aggregate_selection (seq / start_s / busy_s / kv_bytes / flops),
so KV bytes and FLOPs follow the global serving assumptions.

SELECTIONS: every CLICK on a bar makes one, in the next of the 7 vivid
SemiAnalysis brand contrasts (theme.BRAND_PALETTE, cycled the way the
Simulator cycles them). The clicked bin wears that color on its own chart and
the other two charts stack the same color where those requests land - that
stack IS the correlation. Clicking a bin that already belongs to a selection
releases it, and a selection that runs out of bins is dropped, returning its
color to the cycle.

The first pick anchors store['chart'] to that slot; bins are EXCLUSIVE across
selections, and clicks on the other two charts are inert (they are the
answer, not the question). With nothing selected the anchor clears and the
next click may land on any chart.

Store shape:
  {"live": sid|None, "next_sid": int, "next_color": int,
   "chart": "y1"|"y2"|"y3"|None,
   "selections": [{"sid": int, "color": int, "bins": [int, ...]}]}

Every mutation returns a NEW store (copy-on-write). Dash-free module — this
is the unit-test surface.
"""
from __future__ import annotations

from modules.binning import bin_index
from modules.theme import BRAND_PALETTE

# One selection per brand color: every live selection is a
# DIFFERENT color, which is the whole point of the cycle.
MAX_SELECTIONS = len(BRAND_PALETTE)

class InertChart(ValueError):
    """A pick aimed at a chart that is not the selection chart. The caller
    stays silent: the default cursor over that chart already says it."""


class PaletteExhausted(ValueError):
    """Every brand color is in use. The caller must SAY so - a click that
    silently does nothing reads as a broken chart."""

CHART_SLOTS = ("y1", "y2", "y3")

# --- Measure registry (per-chart X measures; y is always request count) ----

MEASURES = {
    "turn_number": dict(
        label="turn # (request seq in conv)",
        info="Bins requests by their sequence number within their "
             "conversation (main and subagent requests both count).",
        getter=lambda p: p["seq"]),
    "cumulative_time": dict(
        label="cumulative time (s)",
        info="Bins requests by wall-clock seconds since their conversation's "
             "first request - idle stretches included.",
        getter=lambda p: p["start_s"]),
    "busy_time": dict(
        label="busy time (s, active only)",
        info="Bins requests by ACTIVE seconds elapsed at their start (time "
             "when at least one request of the conversation was in flight - "
             "idle gaps compressed out).",
        getter=lambda p: p["busy_s"]),
    "kv_cache_tokens": dict(
        label="cached KV at turn start (tokens)",
        info="Input tokens ALREADY in the KV cache when the turn starts (the "
             "prefix-cache hit), distinct from the uncached input the turn "
             "then prefills. Architecture-independent; the byte size is this "
             "times KV bytes/token under the assumptions.",
        getter=lambda p: p["cached_tokens"]),
    "new_input": dict(
        label="uncached input (tokens)",
        info="Uncached input tokens - new text (user message, tool results, "
             "agent hand-offs) not already served from the prompt cache; "
             "this is what prefill actually computes.",
        getter=lambda p: p["uncached_tokens"]),
    "idle_gap": dict(
        label="idle before turn (s)",
        info="Seconds between the end of the previous turn and the start of "
             "this one — the conversation's think/tool/human wait. "
             "Measured from the busy frontier, so a turn starting while a "
             "parallel subagent still runs reads 0. A conversation's FIRST "
             "turn has no previous turn: undefined, so it is left out of "
             "this chart's counts entirely (never binned as zero).",
        getter=lambda p: p["idle_gap_s"]),
    "decode_output": dict(
        label="decode output (tokens)",
        info="Tokens decoded (generated) by the request.",
        getter=lambda p: p["out_tokens"]),
}

DEFAULT_AXES = {"y1": "kv_cache_tokens", "y2": "new_input",
                "y3": "decode_output"}


def measure_values(rows: list[dict], key: str) -> list[float]:
    """One measure over enriched rows. KeyError on unknown key."""
    getter = MEASURES[key]["getter"]
    return [getter(p) for p in rows]


# --- Selection store ---------------------------------------------------------

def selection_color(selection: dict) -> str:
    """The selection's brand color. The index is carried ON the selection, so
    it survives other selections coming and going and never drifts."""
    return BRAND_PALETTE[selection["color"] % len(BRAND_PALETTE)]


def initial_store() -> dict:
    """Nothing selected: no selections, no anchored chart."""
    return {"live": None, "next_sid": 0, "next_color": 0, "chart": None,
            "selections": []}


def _copy(store: dict) -> dict:
    return {"live": store["live"], "next_sid": store["next_sid"],
            "next_color": store.get("next_color", 0),
            "chart": store["chart"],
            "selections": [dict(s, bins=list(s["bins"]))
                           for s in store["selections"]]}


def _take_color(store: dict) -> int:
    """The next free brand color, walking the cycle from where we left off so
    colors rotate instead of always restarting at blue. Raises when all seven
    are in use - with one color per selection there is no honest way to add
    an eighth."""
    c = next_color_index(store)
    if c is None:
        raise PaletteExhausted(
            f"all {len(BRAND_PALETTE)} selection colors are in use — clear "
            f"one before starting another")
    store["next_color"] = (c + 1) % len(BRAND_PALETTE)
    return c


def next_color_index(store: dict) -> int | None:
    """The brand color the next click would take, without taking it - the
    cursor wears this. None when all seven are in use."""
    used = {s["color"] for s in store["selections"]}
    n = len(BRAND_PALETTE)
    start = store.get("next_color", 0) % n
    for step in range(n):
        c = (start + step) % n
        if c not in used:
            return c
    return None


def _drop_empty(store: dict) -> None:
    """A selection with no bins has nothing to show and no reason to hold a
    color."""
    store["selections"] = [s for s in store["selections"] if s["bins"]]
    if store["live"] not in {s["sid"] for s in store["selections"]}:
        store["live"] = None


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
        raise InertChart(
            f"selections live in chart {store['chart']} — this click "
            f"targeted {chart}")


def pick_bin(store: dict, chart: str, bin_idx: int) -> dict:
    """A click on a bar.

    An unowned bin starts a NEW selection in the next brand color; a bin that
    already belongs to a selection is released from it, and that selection is
    dropped if it had nothing else. The first pick anchors the selection
    chart; a click on another chart raises ValueError (the caller renders
    those charts inert).
    """
    out = _copy(store)
    _anchor(out, chart)
    bin_idx = int(bin_idx)
    for s in out["selections"]:
        if bin_idx in s["bins"]:
            s["bins"].remove(bin_idx)
            _drop_empty(out)
            _unanchor_if_all_empty(out)
            return out
    sel = {"sid": out["next_sid"], "color": _take_color(out),
           "bins": [bin_idx]}
    out["next_sid"] += 1
    out["selections"].append(sel)
    out["live"] = sel["sid"]
    return out


def pick_range(store: dict, chart: str, lo_bin: int, hi_bin: int) -> dict:
    """A box-select: ONE gesture, ONE new selection, ONE color, taking the
    inclusive bin range from whoever held those bins."""
    if hi_bin < lo_bin:
        lo_bin, hi_bin = hi_bin, lo_bin
    out = _copy(store)
    _anchor(out, chart)
    claimed = set(range(int(lo_bin), int(hi_bin) + 1))
    for s in out["selections"]:
        s["bins"] = [b for b in s["bins"] if b not in claimed]
    _drop_empty(out)
    sel = {"sid": out["next_sid"], "color": _take_color(out),
           "bins": sorted(claimed)}
    out["next_sid"] += 1
    out["selections"].append(sel)
    out["live"] = sel["sid"]
    return out


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
    _drop_empty(out)
    _unanchor_if_all_empty(out)
    return out


def clear_selection(store: dict, sid: int) -> dict:
    """Drop a selection and return its color to the cycle. The anchor clears
    when nothing is selected any more."""
    out = _copy(store)
    get_selection(out, sid)["bins"] = []
    _drop_empty(out)
    _unanchor_if_all_empty(out)
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
    # None = the measure is undefined for that row (a conversation's first
    # turn has no idle gap): it is a member of nothing, never bin 0
    return [v is not None and bin_index(edges, v, log_x) in bset
            for v in values]


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
