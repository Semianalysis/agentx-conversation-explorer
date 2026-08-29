"""Pure filtering + histogram-binning helpers (the Correlations tab's core).

No Dash imports — this is the unit-test surface.
"""
from __future__ import annotations

import math

_FILTER_KEYS = {"models", "roles"}


def filter_records(records: list[dict], filters: dict) -> tuple[list[dict], dict]:
    """Apply model/role filters. Unknown filter key -> KeyError (typo detection).

    Returns (filtered_records, gate_counts) where gate_counts traces how many
    rows survived each gate: n_pool -> n_model -> n_role.
    """
    unknown = set(filters) - _FILTER_KEYS
    if unknown:
        raise KeyError(f"unknown filter keys {sorted(unknown)}; allowed: {sorted(_FILTER_KEYS)}")
    gates = {"n_pool": len(records)}
    out = records
    models = filters.get("models")
    if models:
        allowed = set(models)
        out = [r for r in out if r["model"] in allowed]
    gates["n_model"] = len(out)
    roles = filters.get("roles")
    if roles:
        allowed = set(roles)
        out = [r for r in out if r["role"] in allowed]
    gates["n_role"] = len(out)
    return out, gates


def make_bins(values: list[float], n_bins: int, log_x: bool) -> list[float]:
    """Bin edges (len n_bins+1) over the value range.

    log_x: geometric edges; zero/negative values are clamped to 1 token for
    binning (documented on the axis). Raises ValueError on empty input —
    the caller decides how to annotate an empty chart, never silently bins [].
    """
    if not values:
        raise ValueError("make_bins on empty values — caller must gate on emptiness")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if log_x:
        lo = max(min(values), 1.0)
        hi = max(max(values), lo)
        if hi == lo:
            hi = lo * 2
        ratio = (hi / lo) ** (1.0 / n_bins)
        edges = [lo * ratio ** i for i in range(n_bins + 1)]
    else:
        lo = float(min(values))
        hi = float(max(values))
        if hi == lo:
            hi = lo + 1
        step = (hi - lo) / n_bins
        edges = [lo + step * i for i in range(n_bins + 1)]
    edges[-1] = edges[-1] * (1 + 1e-9) + 1e-9  # last edge inclusive
    return edges


def bin_index(edges: list[float], value: float, log_x: bool) -> int:
    """Bin index of one value under the same clamping rules as bin_counts:
    values below edge[0] land in bin 0 (log-x clamps sub-1 values), values
    at/above the last edge raise — an out-of-range value means the edges
    don't belong to this data."""
    x = max(value, 1.0) if log_x else value
    if x >= edges[-1]:
        raise ValueError(f"value {value} >= last bin edge {edges[-1]}")
    return max(_bisect(edges, x), 0)


def bin_counts(values: list[float], edges: list[float], log_x: bool) -> list[int]:
    """Histogram counts for precomputed edges (bin_index rules)."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        counts[bin_index(edges, v, log_x)] += 1
    return counts


def bin_weighted(values: list[float], edges: list[float], log_x: bool,
                 weights: list[float]) -> list[float]:
    """Sum of weights per bin (bin_index rules): the 'how much happened in
    this bin' histogram. len(weights) must match len(values)."""
    if len(weights) != len(values):
        raise ValueError(f"{len(weights)} weights for {len(values)} values")
    sums = [0.0] * (len(edges) - 1)
    for v, w in zip(values, weights):
        sums[bin_index(edges, v, log_x)] += w
    return sums


def _bisect(edges: list[float], x: float) -> int:
    lo, hi = 0, len(edges) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if edges[mid] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


def value_stats(values: list[float]) -> dict:
    """count / median / mean / p90 / max for a histogram's annotation."""
    if not values:
        raise ValueError("value_stats on empty values")
    s = sorted(values)
    n = len(s)

    def q(p: float) -> float:
        return s[min(int(p * n), n - 1)]

    return {
        "count": n,
        "median": q(0.5),
        "mean": sum(s) / n,
        "p90": q(0.9),
        "max": s[-1],
    }


def edge_label(edge: float) -> str:
    """Compact tick label for a bin edge."""
    if edge >= 1e6:
        return f"{edge / 1e6:.3g}M"
    if edge >= 1e3:
        return f"{edge / 1e3:.3g}K"
    if edge == int(edge):
        return str(int(edge))
    return f"{edge:.3g}"
