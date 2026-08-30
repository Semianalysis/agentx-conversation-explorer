"""Pure aggregation for the Deep-dive tab: implied compute over a SET of
conversations (the Explorer selection, locally subsettable).

Each conversation has its own clock (start_s is seconds from ITS start), so
timelines overlay on conversation-relative time and wall-clocks SUM (the
traces are independent streams). No Dash imports.
"""
from __future__ import annotations

import math

from modules.arch import DTYPE_BYTES, conversation_compute
from modules.explorer_data import busy_offsets, group_by_conversation


def aggregate_selection(pool: list[dict], conv_ids: list[str], arch: dict,
                        cfg: dict, ordinal: dict[str, int]) -> dict:
    """Aggregate implied compute over conv_ids.

    Returns {'totals' (summed; peak_kv_bytes is the MAX over requests),
             'per_request', 'wall_s_sum', 'n_convs'}.

    per_request rows are ordered by (ordinal, start_s) and enriched for the
    measure registry: 'ord' (conversation ordinal), 'cid', 'seq' (1-based
    request sequence within the conversation, time order), 'cum_flops'
    (running FLOPs within the conversation), 'kv_bytes' (context KV-cache
    bytes under the assumptions), 'busy_s' (ACTIVE seconds elapsed at the
    request's start: the measure of the union of all earlier [start, end]
    request intervals in the conversation — idle gaps contribute nothing).

    Raises on empty conv_ids or a conv_id missing from the pool — an
    aggregate over nothing, or over a phantom conversation, is a caller bug.
    """
    if not conv_ids:
        raise ValueError("aggregate_selection with empty conv_ids")
    by_conv = group_by_conversation(pool)
    missing = [c for c in conv_ids if c not in by_conv]
    if missing:
        raise ValueError(f"conversations not in pool: {missing[:3]}")
    kv_bytes_per_token = (arch["n_layers"] * arch["kv_entries_per_token_per_layer"]
                          * DTYPE_BYTES[cfg["dtype_kv"]])

    totals: dict = {}
    per_request: list[dict] = []
    wall_s_sum = 0.0
    for cid in sorted(conv_ids, key=lambda c: ordinal.get(c, 0)):
        recs = sorted(by_conv[cid], key=lambda r: r["start_s"])
        result = conversation_compute(recs, arch, cfg)
        t = result["totals"]
        if not totals:
            totals = dict(t)
        else:
            for k, v in t.items():
                if k == "peak_kv_bytes":
                    totals[k] = max(totals[k], v)
                else:
                    totals[k] += v
        wall_s_sum += max(r["end_s"] for r in recs) - min(r["start_s"] for r in recs)
        ord_ = ordinal.get(cid, 0)
        cum_flops = 0.0
        cum_out = 0
        # per_request is start-sorted, so busy_offsets applies directly
        busy = busy_offsets(result["per_request"])
        for seq, (p, busy_s) in enumerate(zip(result["per_request"], busy),
                                          start=1):
            cum_flops += p["flops"]
            cum_out += p["out_tokens"]
            per_request.append({
                **p, "ord": ord_, "cid": cid, "seq": seq,
                "cum_flops": cum_flops,
                "cum_out": cum_out,
                "kv_bytes": p["in_tokens"] * kv_bytes_per_token,
                "busy_s": busy_s,
            })
    return {"totals": totals, "per_request": per_request,
            "wall_s_sum": wall_s_sum, "n_convs": len(conv_ids)}


def zoom_member_uids(per_request: list[dict], x_key: str, y_key: str,
                     window_x: list[float], window_y: list[float],
                     x_scale: str) -> set[str]:
    """uids of the requests inside the magnifier window: x measure within
    window_x AND the MAGNIFIED chart's y measure within window_y (both
    windows in axis units — log10 for log axes; y measures are always log).
    The other charts gray out everything not in this set."""
    from modules.measures import ALL_MEASURES, axis_units  # dash-free

    xg = ALL_MEASURES[x_key]["getter"]
    yg = ALL_MEASURES[y_key]["getter"]
    out = set()
    for p in per_request:
        ax = axis_units(xg(p), x_scale)
        ay = axis_units(yg(p), "log")
        if window_x[0] <= ax <= window_x[1] and window_y[0] <= ay <= window_y[1]:
            out.add(p["uid"])
    return out


def grouped_series(per_request: list[dict], x_key: str, y_key: str,
                   n_bins: int = 60) -> dict:
    """Mean / min / max of a y measure ACROSS conversations as a function of x.

    A conversation contributes only where it actually has requests — once it
    reaches its last turn it drops out of the summary entirely (never padded
    or defaulted). min/max are the extreme per-conversation values at each x.

    x='turn_number': exact turn positions (each conversation has at most one
    request per seq). Log-scaled time measures (cumulative_time, busy_time):
    geometric time bins over [1s, max]; a conversation's value in a bin is its
    mean there, and the cross-conversation stats are over those
    per-conversation means.
    x='conv_number' cannot be grouped (each x IS one conversation) -> ValueError.

    Returns {'xs', 'mean', 'lo', 'hi', 'p10', 'p90', 'n_alive'} (aligned lists, xs ascending).
    """
    from modules.measures import ALL_MEASURES  # measures has no dash imports

    if x_key == "conv_number":
        raise ValueError("grouped_series is undefined for x=conv_number "
                         "(each x position is a single conversation)")
    if not per_request:
        raise ValueError("grouped_series on empty per_request")
    xg = ALL_MEASURES[x_key]["getter"]
    yg = ALL_MEASURES[y_key]["getter"]

    # per-conversation value at each x position (mean when several land there)
    acc: dict[float, dict[str, list[float]]] = {}
    if x_key == "turn_number":
        for p in per_request:
            acc.setdefault(xg(p), {}).setdefault(p["cid"], []).append(yg(p))
    else:  # log-scaled time measure -> geometric bins
        xs_all = [xg(p) for p in per_request]
        lo_x, hi_x = 1.0, max(max(xs_all), 1.0)
        if hi_x == lo_x:
            hi_x = lo_x * 2
        log_ratio = math.log(hi_x / lo_x) / n_bins
        for p in per_request:
            x = xg(p)
            b = 0 if x <= lo_x else min(int(math.log(x / lo_x) / log_ratio),
                                        n_bins - 1)
            mid = lo_x * math.exp(log_ratio * (b + 0.5))
            acc.setdefault(mid, {}).setdefault(p["cid"], []).append(yg(p))

    xs, mean, lo, hi, p10, p90, n_alive = [], [], [], [], [], [], []
    for x in sorted(acc):
        conv_means = sorted(sum(v) / len(v) for v in acc[x].values())
        n = len(conv_means)
        xs.append(x)
        mean.append(sum(conv_means) / n)
        lo.append(conv_means[0])
        hi.append(conv_means[-1])
        p10.append(conv_means[min(int(0.10 * n), n - 1)])
        p90.append(conv_means[min(int(0.90 * n), n - 1)])
        n_alive.append(n)
    return {"xs": xs, "mean": mean, "lo": lo, "hi": hi,
            "p10": p10, "p90": p90, "n_alive": n_alive}
