"""Pure data + state helpers for the Correlations tab.

The tab's state is a list of SELECTIONS (at-corr-selections-store). Each
selection owns a set of bins of ONE controlling histogram (its `dim`,
anchored by the first bin picked); once the store is 'applied', the OTHER
histograms draw the conditional distribution of the requests matching those
bins, in the selection's color, stacked across selections. Exactly one
selection is 'live' — bin clicks edit it.

Store shape:
  {"live": sid, "applied": bool, "next_sid": int,
   "selections": [{"sid": int, "dim": str|None, "bins": [int, ...]}]}

Every mutation returns a NEW store (copy-on-write; stores are immutable
archives). All logic here is Dash-free — this is the unit-test surface.
"""
from __future__ import annotations

from modules.records import DIMENSIONS
from modules.theme import PALETTE
from modules.turns_data import bin_index

MAX_SELECTIONS = 10


def selection_color(sid: int) -> str:
    """Stable color for a selection id (survives deletion of other selections)."""
    return PALETTE[sid % len(PALETTE)]


def initial_store() -> dict:
    """One empty live selection, nothing applied."""
    return {"live": 0, "applied": False, "next_sid": 1,
            "selections": [{"sid": 0, "dim": None, "bins": []}]}


def _copy(store: dict) -> dict:
    return {"live": store["live"], "applied": store["applied"],
            "next_sid": store["next_sid"],
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
        if not sel["bins"]:
            sel["dim"] = None
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
    out["selections"].append({"sid": sid, "dim": None, "bins": []})
    out["live"] = sid
    return out


def delete_selection(store: dict, sid: int) -> dict:
    """Remove a selection; the panel always keeps at least one (empty) section."""
    out = _copy(store)
    out["selections"] = [s for s in out["selections"] if s["sid"] != sid]
    if not out["selections"]:
        new_sid = out["next_sid"]
        out["next_sid"] += 1
        out["selections"] = [{"sid": new_sid, "dim": None, "bins": []}]
    if all(s["sid"] != out["live"] for s in out["selections"]):
        out["live"] = out["selections"][-1]["sid"]
    return out


def set_live(store: dict, sid: int) -> dict:
    out = _copy(store)
    get_selection(out, sid)  # KeyError on phantom sid
    out["live"] = sid
    return out


def set_applied(store: dict, applied: bool = True) -> dict:
    out = _copy(store)
    out["applied"] = bool(applied)
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
