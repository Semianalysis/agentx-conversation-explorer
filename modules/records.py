"""Flatten conversation structure trees into immutable per-request records.

Record shape (contract — tested in tests/test_records.py):
  uid            str   conv_id + '#' + tree path ('3', '17.2', ...), unique per dataset
  conv_id        str
  role           'main' | 'subagent'
  agent_id       str | None   (subagent id for nested requests)
  depth          int   0 = main-agent request
  model          str
  in_tokens      int   total input = context length of the request
  cached_tokens  int
  uncached_tokens int  new (uncached) input
  out_tokens     int   decode length
  start_s        float  seconds from conversation start
  end_s          float
  turn_index     int

Invariant enforced at load: in_tokens == cached_tokens + uncached_tokens.
Pure module: no Dash imports. Loaded records are immutable archives — derive,
don't mutate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from modules import api_client

logger = logging.getLogger(__name__)

_TURN_REQUIRED = ("in", "cached", "uncached", "out", "startS", "endS", "model")


def _validate_turn(node: dict, uid: str) -> None:
    missing = [k for k in _TURN_REQUIRED if node.get(k) is None]
    if missing:
        raise ValueError(f"turn node {uid} missing fields {missing}: {json.dumps(node)[:200]}")
    if node["in"] != node["cached"] + node["uncached"]:
        raise ValueError(
            f"turn node {uid} violates in == cached + uncached: "
            f"in={node['in']} cached={node['cached']} uncached={node['uncached']}"
        )
    for k in ("in", "cached", "uncached", "out"):
        if node[k] < 0:
            raise ValueError(f"turn node {uid} has negative {k}={node[k]}")


def _walk(nodes: list[dict], conv_id: str, path: str, role: str,
          agent_id: str | None, depth: int, out: list[dict]) -> None:
    for i, node in enumerate(nodes):
        sub_path = f"{path}.{i}" if path else str(i)
        uid = f"{conv_id}#{sub_path}"
        kind = node.get("kind")
        if kind == "turn":
            _validate_turn(node, uid)
            out.append({
                "uid": uid,
                "conv_id": conv_id,
                "role": role,
                "agent_id": agent_id,
                "depth": depth,
                "model": node["model"],
                "in_tokens": node["in"],
                "cached_tokens": node["cached"],
                "uncached_tokens": node["uncached"],
                "out_tokens": node["out"],
                "start_s": node["startS"],
                "end_s": node["endS"],
                "turn_index": node.get("turnIndex", -1),
            })
        elif kind == "subagent":
            children = node.get("children")
            if not children:
                raise ValueError(f"subagent node {uid} has no children")
            _walk(children, conv_id, sub_path, "subagent",
                  node.get("agentId"), depth + 1, out)
        else:
            raise ValueError(f"unknown node kind {kind!r} at {uid}")


def flatten_conversation(conv_id: str, structure: dict) -> list[dict]:
    """Flatten one conversation's structure tree into per-request records.

    Subagent aggregate nodes are NOT emitted (their children are — emitting both
    would double-count tokens); they only contribute role/agent_id/depth.
    """
    nodes = structure.get("nodes")
    if not nodes:
        raise ValueError(f"conversation {conv_id!r} structure has no nodes")
    records: list[dict] = []
    _walk(nodes, conv_id, "", "main", None, 0, records)
    if not records:
        raise ValueError(f"conversation {conv_id!r}: {len(nodes)} nodes yielded zero request records")
    return records


# --- Server-side record pools (single-user local app) ---

_POOLS: dict[str, list[dict]] = {}


def load_records(slug: str) -> list[dict]:
    """Load + flatten every cached conversation of a dataset. Memoized per slug."""
    if slug in _POOLS:
        return _POOLS[slug]
    conv_dir = api_client.conversation_dir(slug)
    files = sorted(conv_dir.glob("*.json")) if conv_dir.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"no cached conversations for dataset {slug!r} in {conv_dir} — "
            f"load it from the Explorer Dataset panel first"
        )
    records: list[dict] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            conv = json.load(f)
        records.extend(flatten_conversation(conv["conv_id"], conv["structure"]))
    logger.info("loaded %d records from %d conversations for %s", len(records), len(files), slug)
    _POOLS[slug] = records
    return records


NS = "::"   # conv-id namespace separator: "<slug>::<conv_id>"


def namespaced(slug: str, conv_id: str) -> str:
    return f"{slug}{NS}{conv_id}"


def split_conv_id(conv_id: str) -> tuple[str, str]:
    """'<slug>::<id>' -> (slug, id). A bare id (no namespace) raises: every
    multi-dataset path must carry its origin."""
    slug, sep, rest = conv_id.partition(NS)
    if not sep:
        raise ValueError(f"conversation id {conv_id!r} carries no dataset")
    return slug, rest


def load_pool(slugs: list[str]) -> list[dict]:
    """Records of SEVERAL datasets in one pool, conv ids namespaced by slug
    so two datasets can never collide. Empty slug list -> [] (the caller
    renders an empty state; loading nothing is not an error)."""
    out: list[dict] = []
    for slug in slugs or []:
        for r in load_records(slug):
            out.append({**r, "conv_id": namespaced(slug, r["conv_id"]),
                        "uid": namespaced(slug, r["uid"]), "dataset": slug})
    return out


def merged_index(slugs: list[str]) -> list[str]:
    """Namespaced conversation ids of several datasets, each dataset in its
    own token-sorted order - so ordinals stay stable per dataset and unique
    across them."""
    out: list[str] = []
    for slug in slugs or []:
        out += [namespaced(slug, it["conv_id"])
                for it in api_client.fetch_conversation_index(slug)]
    return out


def load_pool_activities(slugs: list[str]) -> dict[str, list[dict]]:
    """Activity intervals of several datasets, keyed by namespaced conv id."""
    out: dict[str, list[dict]] = {}
    for slug in slugs or []:
        for cid, acts in load_activities(slug).items():
            out[namespaced(slug, cid)] = acts
    return out


def clear_pools() -> None:
    _POOLS.clear()
    _ACTIVITIES.clear()


_ACTIVITIES: dict[str, dict[str, list[dict]]] = {}


def load_activities(slug: str) -> dict[str, list[dict]]:
    """{conv_id: [{category, start_s, end_s}]} for datasets whose import
    carried them (OpenClaw exports). AgentX datasets have none, so this is
    {} there and every pause stays honestly 'unknown'. Memoized per slug."""
    if slug in _ACTIVITIES:
        return _ACTIVITIES[slug]
    conv_dir = api_client.conversation_dir(slug)
    out: dict[str, list[dict]] = {}
    for path in sorted(conv_dir.glob("*.json")) if conv_dir.is_dir() else []:
        with open(path, encoding="utf-8") as f:
            conv = json.load(f)
        acts = conv.get("activities")
        if acts:
            out[conv["conv_id"]] = acts
    _ACTIVITIES[slug] = out
    return out


def has_main_agent(recs: list[dict]) -> bool:
    return any(r["role"] == "main" for r in recs)


def lane_turns(recs: list[dict], key=None) -> list[dict]:
    """The conversation's own request lane, ordered.

    Its MAIN-agent requests - or, for a lane that has none, its own requests.
    A lane without a main agent is an OpenClaw subagent lane (AgentX subagents
    nest under a main agent and never produce one), and it is a real
    conversation with a real timeline, so every view summarizes it from the
    requests it does have rather than each deciding separately to drop it.

    key defaults to (turn_index, start_s); pass start_s alone for views that
    care about wall-clock order.
    """
    if not recs:
        raise ValueError("lane_turns on no records")
    key = key or (lambda r: (r["turn_index"], r["start_s"]))
    return sorted(recs if not has_main_agent(recs)
                  else [r for r in recs if r["role"] == "main"], key=key)


def pool_models(records: list[dict]) -> list[str]:
    """Distinct models in a pool, sorted by descending request count."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r["model"]] = counts.get(r["model"], 0) + 1
    return sorted(counts, key=lambda m: -counts[m])


def conversation_ids(records: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for r in records:
        seen.setdefault(r["conv_id"])
    return list(seen)
