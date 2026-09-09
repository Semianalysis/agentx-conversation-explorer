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
        label="cached context at turn start (tokens)", scale="log",
        cumulative=False,
        info="Input tokens ALREADY in the KV cache when the turn starts (the "
             "prefix-cache hit), distinct from the uncached input the turn "
             "then prefills. cached + uncached = the total context attended.",
        getter=lambda p: _clamp1(p["cached_tokens"])),
    "new_input_tokens": dict(
        label="uncached input (tokens, user/tool/agent)", scale="log",
        cumulative=False,
        info="Uncached input tokens — new text (user message, tool results, "
             "agent hand-offs) not already served from the prompt cache; this "
             "is what prefill actually computes.",
        getter=lambda p: _clamp1(p["uncached_tokens"])),
    "new_output_tokens": dict(
        label="decode output per turn (tokens)", scale="log", cumulative=False,
        info="Tokens decoded (generated) by this request.",
        getter=lambda p: _clamp1(p["out_tokens"])),
    "idle_gap": dict(
        label="idle before turn (s)", scale="log", cumulative=False,
        info="Seconds between the end of the previous turn and the start of "
             "this one — the conversation's think/tool/human wait. "
             "Measured from the busy frontier, so a turn that starts while a "
             "parallel subagent is still running reads 0 (no idle interval); "
             "log axis shows 0 at the 1 s floor. The FIRST turn of a "
             "conversation has no previous turn, so it is undefined and "
             "simply not plotted — never faked as zero.",
        getter=lambda p: (None if p["idle_gap_s"] is None
                          else _clamp1(p["idle_gap_s"]))),
    "cumulative_output_tokens": dict(
        label="cumulative decode output (tokens, per conv)", scale="log",
        cumulative=True,
        info="Running total of decoded tokens within the conversation, up to "
             "and including this request. With few conversations selected, "
             "drawn as one line per conversation.",
        getter=lambda p: _clamp1(p["cum_out"])),
}

ALL_MEASURES = {**X_MEASURES, **Y_MEASURES}

DEFAULT_AXES = {"x": "busy_time", "y1": "context_tokens",
                "y2": "new_input_tokens", "y3": "new_output_tokens"}


def measure_series(per_request: list[dict], key: str) -> list[float]:
    """Values of one measure over enriched per-request rows. KeyError on an
    unknown measure key (typo detection)."""
    getter = ALL_MEASURES[key]["getter"]
    return [getter(p) for p in per_request]


def zoom_window(center: float, full_range: list[float],
                factor: float = 5.0) -> list[float]:
    """A window 1/factor the width of full_range, centered on `center` but
    clamped to stay inside it (both in AXIS units: log10 for a log axis).
    factor <= 1 returns the full range unchanged."""
    lo, hi = full_range
    if hi <= lo:
        raise ValueError(f"degenerate range {full_range}")
    w = (hi - lo) / factor
    if w >= hi - lo:
        return [lo, hi]
    c = min(max(center, lo + w / 2), hi - w / 2)
    return [c - w / 2, c + w / 2]


def axis_units(value: float, scale: str) -> float:
    """Data value -> axis units (log10 on a log axis, identity on linear)."""
    if scale == "log":
        if value <= 0:
            raise ValueError(f"log axis cannot place non-positive {value}")
        return math.log10(value)
    return value


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
