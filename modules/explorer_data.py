"""Pure data helpers for the Explorer tab: per-conversation growth curves,
metrics table rows, and the cross-tab conversation-selection filter.

Conversation identity for display is the ORDINAL: the 1-based position in the
dataset's conversation index (API sort=tokens desc, same order the site lists).
conv_id remains the stable lookup key everywhere; the ordinal is a projection.

No Dash imports.
"""
from __future__ import annotations

from modules.arch import (ARCHITECTURES, GPUS, conversation_compute,
                          implied_gpu_seconds)


def group_by_conversation(pool: list[dict]) -> dict[str, list[dict]]:
    by_conv: dict[str, list[dict]] = {}
    for r in pool:
        by_conv.setdefault(r["conv_id"], []).append(r)
    return by_conv


def _main_turns_ordered(conv_records: list[dict]) -> list[dict]:
    return sorted((r for r in conv_records if r["role"] == "main"),
                  key=lambda r: (r["turn_index"], r["start_s"]))


def build_conversation_table(
    pool: list[dict],
    index_conv_ids: list[str],
    models_filter: list[str] | None,
) -> tuple[list[dict], int]:
    """Rows for the conversation list.

    Row: {id (=conv_id, DataTable row id), ordinal, model (first main turn's —
          the conversation's main-agent model), n_turns (main), duration_h
          (start→finish, all requests), isl0 (first main turn's input tokens),
          max_ctx (largest main-turn context — often ABOVE final_ctx because
          compaction shrinks the context), final_ctx (last main turn's input
          tokens), avg_out / max_out (decode tokens per main turn — the mean is
          NOT (final_ctx - isl0)/turns: outputs are consumed by tool calls and
          compaction, not accumulated into context)}.

    models_filter: keep conversations that used ANY of the given models.
    Returns (rows sorted by ordinal, n_skipped_no_main_turns). A conversation
    whose conv_id is missing from the index raises — the index must cover the
    pool (same dataset), anything else is a data bug.
    """
    by_conv = group_by_conversation(pool)
    ordinal = {cid: i + 1 for i, cid in enumerate(index_conv_ids)}
    unknown = set(by_conv) - set(ordinal)
    if unknown:
        raise ValueError(
            f"{len(unknown)} conversations in pool but not in index "
            f"(e.g. {sorted(unknown)[:3]}) — index and pool are from different datasets?")
    wanted_models = set(models_filter) if models_filter else None
    rows: list[dict] = []
    skipped_no_main = 0
    for cid, recs in by_conv.items():
        if wanted_models and not ({r["model"] for r in recs} & wanted_models):
            continue
        main = _main_turns_ordered(recs)
        if not main:
            skipped_no_main += 1
            continue
        rows.append({
            "id": cid,
            "ordinal": ordinal[cid],
            "model": main[0]["model"],
            "n_turns": len(main),
            "duration_h": round(
                (max(r["end_s"] for r in recs) - min(r["start_s"] for r in recs)) / 3600.0,
                2),
            "isl0": main[0]["in_tokens"],
            "max_ctx": max(r["in_tokens"] for r in main),
            "final_ctx": main[-1]["in_tokens"],
            "avg_out": round(sum(r["out_tokens"] for r in main) / len(main), 1),
            "max_out": max(r["out_tokens"] for r in main),
        })
    rows.sort(key=lambda r: r["ordinal"])
    return rows, skipped_no_main


def augment_rows_with_compute(rows: list[dict], pool: list[dict],
                              arch_key: str, gpu_key: str, cfg: dict) -> list[dict]:
    """Add the resolved serving-assumption columns and per-conversation implied
    GPU counts to conversation-table rows (copy-on-write; input rows untouched).

    prefill_gpus / decode_gpus = single-GPU-equivalent busy seconds of that
    phase divided by the conversation's wall-clock: the sustained GPU count
    this trace implies at the assumed MFU/MBU. None (blank cell) when the
    conversation has no measurable wall-clock — never fabricated as 0.
    """
    arch = ARCHITECTURES[arch_key]
    gpu = GPUS[gpu_key]
    by_conv = group_by_conversation(pool)
    out = []
    for row in rows:
        recs = by_conv.get(row["id"])
        if not recs:
            raise ValueError(f"conversation {row['id']!r} in table but not in pool")
        totals = conversation_compute(recs, arch, cfg)["totals"]
        gpu_time = implied_gpu_seconds(totals, gpu, cfg)
        wall_s = max(r["end_s"] for r in recs) - min(r["start_s"] for r in recs)
        out.append({
            **row,
            "gpu": gpu["label"],
            "wq": cfg["dtype_weights"],
            "kvq": cfg["dtype_kv"],
            "tp": cfg["tp"], "pp": cfg["pp"], "dp": cfg["dp"],
            # 3 significant figures, NOT fixed decimals — a small sustained
            # count (0.0012 GPUs) must never round down to a fabricated 0.
            "prefill_gpus": _sig3(gpu_time["prefill_s"] / wall_s) if wall_s > 0 else None,
            "decode_gpus": _sig3(gpu_time["decode_s"] / wall_s) if wall_s > 0 else None,
        })
    return out


def _sig3(x: float) -> float:
    return float(f"{x:.3g}")


def conversation_curves(pool: list[dict], conv_ids: set[str] | None = None,
                        ) -> dict[str, tuple[list[int], list[int]]]:
    """Growth curve per conversation: x = main-turn ordinal (1..n, linear),
    y = that turn's input context tokens (log axis on the chart).

    conv_ids limits which conversations get curves (None = all in pool).
    Conversations with zero main turns are skipped (no curve to draw).
    """
    curves: dict[str, tuple[list[int], list[int]]] = {}
    for cid, recs in group_by_conversation(pool).items():
        if conv_ids is not None and cid not in conv_ids:
            continue
        main = _main_turns_ordered(recs)
        if not main:
            continue
        xs = list(range(1, len(main) + 1))
        ys = [r["in_tokens"] for r in main]
        curves[cid] = (xs, ys)
    return curves


def apply_selection(records: list[dict], selection: dict | None, slug: str,
                    ) -> tuple[list[dict], dict]:
    """Restrict a record pool to the Explorer's selected conversations.

    selection = {'slug': ..., 'conv_ids': [...]} (the cross-tab selection
    store). Applied only when it matches this pool's dataset; a selection made
    on another dataset is ignored rather than silently zeroing the rows.
    Returns (records, gate_counts_update).
    """
    if not selection or not selection.get("conv_ids"):
        return records, {}
    if selection.get("slug") != slug:
        return records, {}
    ids = set(selection["conv_ids"])
    out = [r for r in records if r["conv_id"] in ids]
    return out, {"n_sel_convs": len(ids), "n_after_selection": len(out)}


def sync_updates(values: list, trig_index: int) -> tuple | None:
    """Propagate one widget's value across a group of synced widgets (the
    shared dataset dropdowns: every tab is a viewport onto the SAME dataset).

    Returns (value, stale_indices) — the trigger's value and the widgets that
    must be written — or None when all values already agree; the caller must
    then skip the write, or the sync echoes forever. The value itself may be
    None (a cleared dropdown clears the others too), which is why "no change"
    is signaled by index absence, never by a None value.
    """
    if not 0 <= trig_index < len(values):
        raise IndexError(f"trig_index {trig_index} out of range 0..{len(values) - 1}")
    v = values[trig_index]
    stale = [i for i, x in enumerate(values) if x != v]
    if not stale:
        return None
    return v, stale
