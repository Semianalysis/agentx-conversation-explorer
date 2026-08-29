"""Measure registry for the deep-dive chart matrix.

Every chart plots the per-request records of the selected conversations
(enriched by deepdive_data.aggregate_selection). A measure = axis label +
scale + getter over one enriched per-request dict.

Scale rule (user spec): ordinals (conversation number, turn number) are
LINEAR; every magnitude (time, tokens, bytes, FLOPs) is LOG, with values
clamped UP to 1 unit (1 s / 1 token / 1 byte / 1 FLOP) so sub-unit and zero
values are kept on the chart at the axis floor rather than dropped.

Pure module: no Dash imports.
"""
from __future__ import annotations

import math


def _clamp1(v: float) -> float:
    return v if v > 1.0 else 1.0


X_MEASURES = {
    "conv_number": dict(
        label="conversation #", scale="linear",
        info="Conversation ordinal — its rank in the dataset's token-sorted "
             "index (1 = most total input tokens). Same numbering as the '#' "
             "column of the conversation list. Linear axis (it's an ordinal, "
             "not a magnitude).",
        getter=lambda p: p["ord"]),
    "turn_number": dict(
        label="turn # (request seq in conv)", scale="linear",
        info="Request sequence number within its conversation, in start-time "
             "order. Main-agent and subagent requests both count. Linear axis.",
        getter=lambda p: p["seq"]),
    "cumulative_time": dict(
        label="cumulative time (s, ≥1)", scale="log",
        info="Wall-clock seconds since the conversation's first request, at "
             "this request's start. INCLUDES idle stretches when no request "
             "was running (human think time, nights, weekends — often >80% of "
             "a conversation's span), which appear as horizontal gaps. Log "
             "axis; values under 1 s are shown at 1.",
        getter=lambda p: _clamp1(p["start_s"])),
    "busy_time": dict(
        label="busy time (s, active only, ≥1)", scale="log",
        info="ACTIVE seconds elapsed at this request's start — time when at "
             "least one request of the conversation (main or subagent) was in "
             "flight. Idle gaps are compressed out, so this axis shows pure "
             "serving activity; compare with cumulative time to see the idle "
             "gaps. Log axis; values under 1 s are shown at 1.",
        getter=lambda p: _clamp1(p["busy_s"])),
}

Y_MEASURES = {
    "context_tokens": dict(
        label="context size (tokens)", scale="log", cumulative=False,
        info="Total input tokens of the request (cached + new) — the context "
             "the model attends over.",
        getter=lambda p: _clamp1(p["in_tokens"])),
    "context_kv_bytes": dict(
        label="context KV-cache size (bytes)", scale="log", cumulative=False,
        info="Bytes of KV cache the request's context occupies under the "
             "selected architecture and KV precision (layers × KV entries per "
             "token × bytes per entry).",
        getter=lambda p: _clamp1(p["kv_bytes"])),
    "new_input_tokens": dict(
        label="new input (uncached tokens, user/tool/agent)", scale="log",
        cumulative=False,
        info="Uncached input tokens — new text (user message, tool results, "
             "agent hand-offs) not already served from the prompt cache; this "
             "is what prefill actually computes.",
        getter=lambda p: _clamp1(p["uncached_tokens"])),
    "new_output_tokens": dict(
        label="decode output per turn (tokens)", scale="log", cumulative=False,
        info="Tokens decoded (generated) by this request.",
        getter=lambda p: _clamp1(p["out_tokens"])),
    "cumulative_output_tokens": dict(
        label="cumulative decode output (tokens, per conv)", scale="log",
        cumulative=True,
        info="Running total of decoded tokens within the conversation, up to "
             "and including this request. With few conversations selected, "
             "drawn as one line per conversation.",
        getter=lambda p: _clamp1(p["cum_out"])),
    "cumulative_flops": dict(
        label="cumulative FLOPs (per conv)", scale="log", cumulative=True,
        info="Running total of implied FLOPs (prefill + decode) within the "
             "conversation under the selected architecture — implied by token "
             "counts, not measured.",
        getter=lambda p: _clamp1(p["cum_flops"])),
    "current_flops": dict(
        label="turn FLOPs", scale="log", cumulative=False,
        info="Implied FLOPs of this single request (prefill + decode) under "
             "the selected architecture.",
        getter=lambda p: _clamp1(p["flops"])),
}

ALL_MEASURES = {**X_MEASURES, **Y_MEASURES}

DEFAULT_AXES = {"x": "cumulative_time", "y1": "context_tokens",
                "y2": "cumulative_flops", "y3": "current_flops"}


def measure_series(per_request: list[dict], key: str) -> list[float]:
    """Values of one measure over enriched per-request rows. KeyError on an
    unknown measure key (typo detection)."""
    getter = ALL_MEASURES[key]["getter"]
    return [getter(p) for p in per_request]


def shared_axis_range(values: list[float], scale: str) -> list[float]:
    """Explicit plotly axis range covering `values`, padded ~2% per side so
    extreme points stay visible. Locks the three deep-dive charts to one
    identical x window (independent autorange pads marker and line traces
    differently, misaligning them). Log scale -> the pair is in log10 units
    (plotly's convention); raises on empty values or non-positive log input.
    """
    if not values:
        raise ValueError("shared_axis_range over no values")
    lo, hi = min(values), max(values)
    if scale == "log":
        if lo <= 0:
            raise ValueError(f"log-axis range needs positive values, got min={lo}")
        log_lo, log_hi = math.log10(lo), math.log10(hi)
        pad = max((log_hi - log_lo) * 0.02, 0.05)
        return [log_lo - pad, log_hi + pad]
    pad = max((hi - lo) * 0.02, 0.5)
    return [lo - pad, hi + pad]
