"""Flatten RAW cc-traces-weka conversations (HuggingFace rows) into the same
`structure` the AgentX API serves — a faithful Python port of the site's own
ingest (InferenceX-app packages/db/src/etl/weka-structure.ts, fetched
2026-08-29), so datasets imported straight from HF are bit-compatible with
API-imported ones.

Semantics ported exactly:
- cached/uncached split = contiguous LEADING hash_ids already seen under an
  infinite KV cache (aiperf _count_seen_prefix_blocks); after counting, ALL
  of the request's blocks join `seen`; cached is clamped to the request's
  `in` (the last block may be partial).
- main turns share one accumulating `seen`; each subagent group runs against
  a COPY of the parent's set at spawn and is never folded back.
- timing: startS = t, endS = t + api_time; a subagent child whose t is
  smaller than the group's t is legacy-RELATIVE and gets group.t added.

Pure module: no Dash, no network.
"""
from __future__ import annotations

DEFAULT_BLOCK_SIZE = 64


def count_seen_prefix_blocks(hash_ids: list[int], seen: set[int]) -> int:
    hits = 0
    for h in hash_ids:
        if h not in seen:
            break
        hits += 1
    return hits


def _split_input(req: dict, seen: set[int], block_size: int) -> tuple[int, int, int]:
    """(in, cached, uncached) for one request; folds its blocks into seen."""
    input_tokens = max(0, round(req.get("in") or 0))
    hash_ids = req.get("hash_ids") or []
    if not hash_ids:
        return input_tokens, 0, input_tokens
    cached_blocks = count_seen_prefix_blocks(hash_ids, seen)
    seen.update(hash_ids)
    cached = min(input_tokens, cached_blocks * block_size)
    return input_tokens, cached, input_tokens - cached


def _finite_time(value) -> float | None:
    if isinstance(value, (int, float)) and value >= 0 and value == value:
        return float(value)
    return None


def _subagent_request_start_s(entry: dict, request: dict) -> float | None:
    request_start = _finite_time(request.get("t"))
    if request_start is None:
        return None
    group_start = _finite_time(entry.get("t"))
    if group_start is not None and request_start + 1e-6 < group_start:
        return group_start + request_start  # legacy-relative child timestamp
    return request_start


def _end_s(start_s: float | None, api_time) -> float | None:
    if start_s is None:
        return None
    return start_s + (_finite_time(api_time) or 0.0)


def _subagent_time_range(entry: dict) -> tuple[float | None, float | None]:
    children = entry.get("requests") or []
    child_starts = [s for c in children
                    if (s := _subagent_request_start_s(entry, c)) is not None]
    start_s = _finite_time(entry.get("t"))
    if start_s is None and child_starts:
        start_s = min(child_starts)
    duration_ms = _finite_time(entry.get("duration_ms"))
    if start_s is not None and duration_ms is not None:
        return start_s, start_s + duration_ms / 1000.0
    child_ends = [s + (_finite_time(c.get("api_time")) or 0.0)
                  for c in children
                  if (s := _subagent_request_start_s(entry, c)) is not None]
    return start_s, (max(child_ends) if child_ends else start_s)


def build_conversation_structure(conv: dict,
                                 block_size_override: int | None = None) -> dict:
    """Raw HF conversation row -> {'blockSize', 'nodes', 'totals'} exactly as
    the AgentX API serves it. Raises on missing 'requests' (malformed row)."""
    if not isinstance(conv.get("requests"), list):
        raise ValueError(f"raw conversation {conv.get('id')!r} has no requests[]")
    block_size = (block_size_override or conv.get("block_size")
                  or DEFAULT_BLOCK_SIZE)
    seen: set[int] = set()
    nodes: list[dict] = []
    totals = dict.fromkeys(("in", "out", "cached", "uncached"), 0)
    num_turns = num_groups = turn_index = 0

    for idx, entry in enumerate(conv["requests"]):
        if entry.get("type") == "subagent":
            g_start, g_end = _subagent_time_range(entry)
            child_seen = set(seen)  # snapshot at spawn; never merged back
            children = []
            g = dict.fromkeys(("in", "out", "cached", "uncached"), 0)
            for inner_idx, inner in enumerate(entry.get("requests") or []):
                in_t, cached, uncached = _split_input(inner, child_seen,
                                                      block_size)
                out = max(0, round(inner.get("out") or 0))
                c_start = _subagent_request_start_s(entry, inner)
                children.append({
                    "kind": "turn", "turnIndex": turn_index,
                    "rawIndex": idx, "innerIndex": inner_idx,
                    "startS": c_start,
                    "endS": _end_s(c_start, inner.get("api_time")),
                    "model": inner.get("model"),
                    "in": in_t, "out": out,
                    "cached": cached, "uncached": uncached,
                })
                turn_index += 1
                for k, v in (("in", in_t), ("out", out), ("cached", cached),
                             ("uncached", uncached)):
                    g[k] += v
            label = (entry.get("subagent_type") or "").strip() or "Subagent"
            nodes.append({
                "kind": "subagent", "label": label,
                "agentId": entry.get("agent_id"), "rawIndex": idx,
                "startS": g_start, "endS": g_end,
                "durationMs": entry.get("duration_ms"),
                **g, "children": children,
            })
            num_groups += 1
            for k in totals:
                totals[k] += g[k]
        else:
            in_t, cached, uncached = _split_input(entry, seen, block_size)
            out = max(0, round(entry.get("out") or 0))
            start_s = _finite_time(entry.get("t"))
            nodes.append({
                "kind": "turn", "turnIndex": turn_index, "rawIndex": idx,
                "startS": start_s,
                "endS": _end_s(start_s, entry.get("api_time")),
                "model": entry.get("model"),
                "in": in_t, "out": out,
                "cached": cached, "uncached": uncached,
            })
            turn_index += 1
            num_turns += 1
            for k, v in (("in", in_t), ("out", out), ("cached", cached),
                         ("uncached", uncached)):
                totals[k] += v

    return {"blockSize": block_size, "nodes": nodes,
            "totals": {**totals, "numTurns": num_turns,
                       "numSubagentGroups": num_groups}}
