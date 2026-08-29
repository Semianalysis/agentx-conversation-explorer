"""Pure data helpers for the Correlations tab.

Condition on a range of one dimension; return the values of every dimension for
the surviving subset. No Dash imports.
"""
from __future__ import annotations

from modules.records import DIMENSIONS


def conditional_subset(
    records: list[dict],
    cond_dim: str,
    lo_tokens: float | None,
    hi_tokens: float | None,
) -> tuple[list[dict], dict]:
    """Records whose cond_dim value lies in [lo_tokens, hi_tokens] (inclusive;
    None = unbounded on that side). Unknown cond_dim -> KeyError.

    Returns (subset, gate_counts): n_pool -> n_in_range.
    """
    field = DIMENSIONS[cond_dim]["field"]
    gates = {"n_pool": len(records)}
    lo = float("-inf") if lo_tokens is None else float(lo_tokens)
    hi = float("inf") if hi_tokens is None else float(hi_tokens)
    if lo > hi:
        raise ValueError(f"conditional range inverted: lo={lo_tokens} > hi={hi_tokens}")
    subset = [r for r in records if lo <= r[field] <= hi]
    gates["n_in_range"] = len(subset)
    return subset, gates
