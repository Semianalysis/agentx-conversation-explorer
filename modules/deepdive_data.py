"""Pure aggregation for the Deep-dive tab: implied compute over a SET of
conversations (the Explorer selection, locally subsettable).

Each conversation has its own clock (start_s is seconds from ITS start), so
timelines overlay on conversation-relative time and wall-clocks SUM (the
traces are independent streams). No Dash imports.
"""
from __future__ import annotations

import math

from modules.arch import DTYPE_BYTES, conversation_compute
from modules.explorer_data import group_by_conversation


def aggregate_selection(pool: list[dict], conv_ids: list[str], arch: dict,
                        cfg: dict, ordinal: dict[str, int]) -> dict:
    """Aggregate implied compute over conv_ids.

    Returns {'totals' (summed; peak_kv_bytes is the MAX over requests),
             'per_request', 'wall_s_sum', 'n_convs'}.

    per_request rows are ordered by (ordinal, start_s) and enriched for the
    measure registry: 'ord' (conversation ordinal), 'cid', 'seq' (1-based
    request sequence within the conversation, time order), 'cum_flops'
    (running FLOPs within the conversation), 'kv_bytes' (context KV-cache
    bytes under the assumptions).

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
        for seq, p in enumerate(result["per_request"], start=1):
            cum_flops += p["flops"]
            cum_out += p["out_tokens"]
            per_request.append({
                **p, "ord": ord_, "cid": cid, "seq": seq,
                "cum_flops": cum_flops,
                "cum_out": cum_out,
                "kv_bytes": p["in_tokens"] * kv_bytes_per_token,
            })
    return {"totals": totals, "per_request": per_request,
            "wall_s_sum": wall_s_sum, "n_convs": len(conv_ids)}


def grouped_series(per_request: list[dict], x_key: str, y_key: str,
                   n_bins: int = 60) -> dict:
    """Mean / min / max of a y measure ACROSS conversations as a function of x.

    A conversation contributes only where it actually has requests — once it
    reaches its last turn it drops out of the summary entirely (never padded
    or defaulted). min/max are the extreme per-conversation values at each x.

    x='turn_number': exact turn positions (each conversation has at most one
    request per seq). x='cumulative_time': geometric time bins over [1s, max];
    a conversation's value in a bin is its mean there, and the cross-
    conversation stats are over those per-conversation means.
    x='conv_number' cannot be grouped (each x IS one conversation) -> ValueError.

    Returns {'xs', 'mean', 'lo', 'hi', 'n_alive'} (aligned lists, xs ascending).
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
    else:  # cumulative_time -> geometric bins
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

    xs, mean, lo, hi, n_alive = [], [], [], [], []
    for x in sorted(acc):
        conv_means = [sum(v) / len(v) for v in acc[x].values()]
        xs.append(x)
        mean.append(sum(conv_means) / len(conv_means))
        lo.append(min(conv_means))
        hi.append(max(conv_means))
        n_alive.append(len(conv_means))
    return {"xs": xs, "mean": mean, "lo": lo, "hi": hi, "n_alive": n_alive}
