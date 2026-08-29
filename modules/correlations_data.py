"""Pure data + state helpers for the Correlations tab.

The tab's state is a list of SELECTIONS (at-corr-selections-store). Each
selection owns a set of bins of ONE controlling histogram (its `dim`,
anchored by the first bin picked) and its own apply toggle `on`: while on,
the OTHER histograms draw the conditional distribution of the requests
matching its bins, in the selection's color, stacked across selections.
Exactly one selection is 'live' — bin clicks edit it. Selections never
disappear on their own: at least one section always exists, and clearing
empties a section rather than removing it.

Store shape:
  {"live": sid, "next_sid": int,
   "selections": [{"sid": int, "dim": str|None, "bins": [int, ...],
                   "on": bool}]}

Every mutation returns a NEW store (copy-on-write; stores are immutable
archives). All logic here is Dash-free — this is the unit-test surface.
"""
from __future__ import annotations

from modules.records import DIMENSIONS
from modules.theme import PALETTE
from modules.binning import bin_index

MAX_SELECTIONS = 10


def selection_color(sid: int) -> str:
    """Stable color for a selection id (survives deletion of other selections)."""
    return PALETTE[sid % len(PALETTE)]


def _empty_selection(sid: int) -> dict:
    return {"sid": sid, "dim": None, "bins": [], "on": False}


def initial_store() -> dict:
    """One empty live selection, apply off."""
    return {"live": 0, "next_sid": 1, "selections": [_empty_selection(0)]}


def _copy(store: dict) -> dict:
    return {"live": store["live"], "next_sid": store["next_sid"],
            "selections": [dict(s, bins=list(s["bins"]))
                           for s in store["selections"]]}


def get_selection(store: dict, sid: int) -> dict:
    for s in store["selections"]:
        if s["sid"] == sid:
            return s
    raise KeyError(f"no selection with sid={sid}")


def _anchor(sel: dict, dim: str) -> None:
    if dim not in DIMENSIONS:
        raise KeyError(f"unknown dimension {dim!r}")
    if sel["dim"] is None:
        sel["dim"] = dim
    if sel["dim"] != dim:
        raise ValueError(
            f"selection {sel['sid']} controls '{sel['dim']}' — click that "
            f"histogram, or add a New selection for '{dim}'")


def toggle_bin(store: dict, sid: int, dim: str, bin_idx: int) -> dict:
    """Toggle one bin on selection sid. An empty selection anchors to `dim`
    with its first bin; a click on a different dim than the anchor raises
    ValueError (the caller shows the message as a hint). Removing the last
    bin un-anchors the selection so it can be re-aimed."""
    out = _copy(store)
    sel = get_selection(out, sid)
    _anchor(sel, dim)
    if bin_idx in sel["bins"]:
        sel["bins"].remove(bin_idx)
        if not sel["bins"]:  # empty again: un-anchored, apply off
            sel["dim"] = None
            sel["on"] = False
    else:
        sel["bins"] = sorted(sel["bins"] + [int(bin_idx)])
    return out


def add_bin_range(store: dict, sid: int, dim: str, lo_bin: int, hi_bin: int) -> dict:
    """Box-select: ADD the inclusive bin range (same anchoring rule)."""
    if hi_bin < lo_bin:
        lo_bin, hi_bin = hi_bin, lo_bin
    out = _copy(store)
    sel = get_selection(out, sid)
    _anchor(sel, dim)
    sel["bins"] = sorted(set(sel["bins"]) | set(range(int(lo_bin), int(hi_bin) + 1)))
    return out


def add_selection(store: dict) -> dict:
    """Append a new empty selection and make it live."""
    out = _copy(store)
    if len(out["selections"]) >= MAX_SELECTIONS:
        raise ValueError(f"at most {MAX_SELECTIONS} selections")
    sid = out["next_sid"]
    out["next_sid"] += 1
    out["selections"].append(_empty_selection(sid))
    out["live"] = sid
    return out


def clear_selection(store: dict, sid: int) -> dict:
    """Empty a selection (bins gone, un-anchored, apply off). The section
    itself stays — selections never disappear, they just go empty."""
    out = _copy(store)
    sel = get_selection(out, sid)
    sel["dim"] = None
    sel["bins"] = []
    sel["on"] = False
    return out


def toggle_on(store: dict, sid: int) -> dict:
    """Flip one selection's apply toggle (conditioning the other charts)."""
    out = _copy(store)
    sel = get_selection(out, sid)
    sel["on"] = not sel["on"]
    return out


def set_live(store: dict, sid: int) -> dict:
    out = _copy(store)
    get_selection(out, sid)  # KeyError on phantom sid
    out["live"] = sid
    return out


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


def bins_subset(records: list[dict], dim: str, edges: list[float],
                bins: list[int], log_x: bool) -> tuple[list[dict], dict]:
    """Records whose `dim` value falls in any of the selected bin indices
    (bin membership follows turns_data.bin_index, so it matches what the
    histogram drew). Empty bins or out-of-range indices raise — conditioning
    on nothing, or on bins the chart never had, is a caller bug.

    Returns (subset, gate_counts): n_pool -> n_matched.
    """
    if not bins:
        raise ValueError("bins_subset with no bins selected")
    n = len(edges) - 1
    bad = [b for b in bins if not 0 <= b < n]
    if bad:
        raise ValueError(f"bin indices out of range 0..{n - 1}: {bad}")
    field = DIMENSIONS[dim]["field"]
    bset = set(bins)
    subset = [r for r in records if bin_index(edges, r[field], log_x) in bset]
    return subset, {"n_pool": len(records), "n_matched": len(subset)}
