"""Pure pause model for the Pause Analytics tab.

A PAUSE is dead air on a conversation's main-agent lane: the span between
the moment the lane last finished work and the moment its next request
starts. Each pause is decomposed EXACTLY into causes:

  subagent            a nested agent was running (AgentX records this in the
                      trace structure, so it is known even though AgentX
                      records nothing else about pauses)
  <tool category>     an imported activity interval covered it (OpenClaw
                      exports carry shell_code / web / files / ... )
  unknown             nothing in the data accounts for it - human think
                      time, queueing, harness overhead. NEVER guessed.

The decomposition is a partition: sum(causes) + unknown == duration to the
microsecond (asserted), so stacked bars always add up to the bar's width in
time and 'unknown' is the honest remainder rather than a modelling fudge.

No Dash imports - this is the unit-test surface.
"""
from __future__ import annotations

import math

from modules.records import has_main_agent, lane_turns

UNKNOWN = "unknown"
SUBAGENT = "subagent"
_EPS = 1e-6


def _union_len(intervals: list[tuple[float, float]]) -> float:
    """Measure of the union of [start, end) intervals (overlaps counted once)."""
    out = 0.0
    hi = -math.inf
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if s > hi:
            out += e - s
            hi = e
        elif e > hi:
            out += e - hi
            hi = e
    return out


def decompose(a: float, b: float,
              labelled: list[tuple[str, float, float]]) -> dict[str, float]:
    """Partition [a, b) across labels. Earlier labels in `labelled` win any
    overlap, so a second cause never double-counts seconds already claimed.
    Returns {label: seconds, 'unknown': seconds}; the values sum to b - a.

    Raises on an inverted span - a caller producing b < a has a bug.
    """
    span = b - a
    if span < 0:
        raise ValueError(f"inverted pause span: [{a}, {b})")
    out: dict[str, float] = {}
    claimed: list[tuple[float, float]] = []
    for label, s, e in labelled:
        s, e = max(s, a), min(e, b)
        if e <= s:
            continue
        # seconds this label adds beyond everything already claimed
        before = _union_len(claimed)
        claimed.append((s, e))
        gained = _union_len(claimed) - before
        if gained > 0:
            out[label] = out.get(label, 0.0) + gained
    covered = sum(out.values())
    out[UNKNOWN] = max(span - covered, 0.0)
    total = sum(out.values())
    assert abs(total - span) < _EPS, (
        f"pause decomposition lost {span - total:.9f}s of [{a}, {b})")
    return out


def conversation_pauses(records: list[dict],
                        activities: list[dict] | None = None) -> list[dict]:
    """Pauses of ONE conversation's main-agent lane, each with the request
    that ended it and its cause breakdown.

    records: that conversation's per-request rows (role/start_s/end_s/...).
    activities: optional [{category, start_s, end_s}] from an import.

    A conversation whose requests are ALL subagent ones (an OpenClaw
    subagent lane) uses those requests as its lane - otherwise it would have
    no timeline at all. Returns [] when the lane has fewer than 2 requests
    (nothing to pause between); never fabricates a zero-length pause.
    """
    if not records:
        raise ValueError("conversation_pauses on no records")
    lane = lane_turns(records, key=lambda r: r["start_s"])
    off_lane = ([r for r in records if r["role"] != "main"]
                if has_main_agent(records) else [])
    acts = activities or []

    out = []
    frontier = lane[0]["end_s"]
    for prev, nxt in zip(lane, lane[1:]):
        frontier = max(frontier, prev["end_s"])
        a, b = frontier, nxt["start_s"]
        if b - a <= 0:            # back-to-back or overlapping: no pause
            continue
        labelled = [(SUBAGENT, r["start_s"], r["end_s"]) for r in off_lane]
        labelled += [(x["category"], x["start_s"], x["end_s"]) for x in acts]
        causes = decompose(a, b, labelled)
        out.append({
            "cid": nxt["conv_id"], "start_s": a, "end_s": b,
            "duration_s": b - a,
            "next_uid": nxt["uid"], "next_model": nxt["model"],
            "next_in_tokens": nxt["in_tokens"],
            "next_cached_tokens": nxt["cached_tokens"],
            "next_uncached_tokens": nxt["uncached_tokens"],
            "next_out_tokens": nxt["out_tokens"],
            "causes": causes,
        })
    return out


def all_pauses(pool: list[dict], conv_ids: set[str] | None = None,
               activities: dict[str, list[dict]] | None = None) -> list[dict]:
    """Pauses across a record pool, conversation by conversation."""
    by_conv: dict[str, list[dict]] = {}
    for r in pool:
        if conv_ids is None or r["conv_id"] in conv_ids:
            by_conv.setdefault(r["conv_id"], []).append(r)
    out = []
    for cid, recs in by_conv.items():
        out.extend(conversation_pauses(recs, (activities or {}).get(cid)))
    return out


def cause_totals(pauses: list[dict]) -> dict[str, float]:
    """Seconds per cause over every pause, descending, 'unknown' last."""
    tot: dict[str, float] = {}
    for p in pauses:
        for k, v in p["causes"].items():
            tot[k] = tot.get(k, 0.0) + v
    known = {k: v for k, v in tot.items() if k != UNKNOWN and v > 0}
    out = dict(sorted(known.items(), key=lambda kv: -kv[1]))
    if tot.get(UNKNOWN):
        out[UNKNOWN] = tot[UNKNOWN]
    return out


def duration_bins(pauses: list[dict], n_bins: int, log_x: bool) -> list[float]:
    """Shared duration bin edges for every chart on the tab, so the stacked
    causes and the ISL chart line up on one x axis. Raises on no pauses -
    the caller gates on emptiness and says so."""
    if not pauses:
        raise ValueError("duration_bins on no pauses")
    from modules.binning import make_bins
    return make_bins([p["duration_s"] for p in pauses], n_bins, log_x)


def stack_by_cause(pauses: list[dict], edges: list[float], log_x: bool,
                   causes: list[str], weight: str = "time",
                   ) -> dict[str, list[float]]:
    """Per-bin height of each cause: 'time' = seconds of that cause inside
    pauses of that duration, 'count' = pauses of that duration whose LARGEST
    cause is that one. Unknown weight -> ValueError (typo detection)."""
    if weight not in ("time", "count"):
        raise ValueError(f"unknown weight {weight!r}; use 'time' or 'count'")
    from modules.binning import bin_index
    n = len(edges) - 1
    out = {c: [0.0] * n for c in causes}
    for p in pauses:
        i = bin_index(edges, p["duration_s"], log_x)
        if weight == "time":
            for c in causes:
                out[c][i] += p["causes"].get(c, 0.0)
        else:
            top = max(p["causes"].items(), key=lambda kv: kv[1])[0]
            if top in out:
                out[top][i] += 1
    return out


def isl_by_bin(pauses: list[dict], edges: list[float], log_x: bool,
               field: str) -> dict[str, list[float | None]]:
    """Median / p10 / p90 of the ISL delivered by the request that ENDS each
    pause, per duration bin. Empty bins yield None (dropped by the chart),
    never 0 - a bin with no pauses has no median."""
    from modules.binning import bin_index
    n = len(edges) - 1
    buckets: list[list[float]] = [[] for _ in range(n)]
    for p in pauses:
        buckets[bin_index(edges, p["duration_s"], log_x)].append(p[field])
    med: list[float | None] = []
    lo: list[float | None] = []
    hi: list[float | None] = []
    for vals in buckets:
        if not vals:
            med.append(None), lo.append(None), hi.append(None)
            continue
        v = sorted(vals)
        med.append(v[len(v) // 2])
        lo.append(v[min(int(0.10 * len(v)), len(v) - 1)])
        hi.append(v[min(int(0.90 * len(v)), len(v) - 1)])
    return {"median": med, "p10": lo, "p90": hi,
            "n": [float(len(b)) for b in buckets]}
