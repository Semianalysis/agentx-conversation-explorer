"""Model-architecture and GPU presets + implied compute/memory/network math.

Claude model internals are not public, so the deep-dive pairs an observed trace
with a USER-CHOSEN architecture preset (generic dense / MoE shapes at several
scales) and a GPU preset. All outputs are order-of-magnitude engineering
estimates from standard transformer roofline formulas; each formula is stated
in its docstring.

Conventions:
  P_active = parameters touched per token (== total for dense).
  C = cached context tokens at request start, U = uncached (new) input tokens,
  O = output (decode) tokens, ctx_in = C + U = in_tokens.
  L = n_layers, Hq = query heads, dh = head dim.

Pure module: no Dash imports. Unit-named fields throughout (_b = count in
units, _bytes, _flops, _s, _gb, _tbps).
"""
from __future__ import annotations

# --- Architecture presets ---------------------------------------------------
# kv_entries_per_token_per_layer: stored K+V scalars per token per layer.
#   GQA default = 2 * n_kv_heads * d_head; MLA (DeepSeek-style) stores one
#   compressed latent (~576 scalars) instead.

ARCHITECTURES = {
    "dense-70b": dict(
        label="Dense 70B (Llama-3-70B-like)",
        params_total=70e9, params_active=70e9,
        n_layers=80, d_model=8192, n_q_heads=64, n_kv_heads=8, d_head=128,
        kv_entries_per_token_per_layer=2 * 8 * 128,
        n_experts=0, topk=0,
    ),
    "dense-405b": dict(
        label="Dense 405B (Llama-3-405B-like)",
        params_total=405e9, params_active=405e9,
        n_layers=126, d_model=16384, n_q_heads=128, n_kv_heads=8, d_head=128,
        kv_entries_per_token_per_layer=2 * 8 * 128,
        n_experts=0, topk=0,
    ),
    "moe-671b-a37b": dict(
        label="MoE 671B / 37B active (DeepSeek-V3-like, MLA)",
        params_total=671e9, params_active=37e9,
        n_layers=61, d_model=7168, n_q_heads=128, n_kv_heads=128, d_head=128,
        kv_entries_per_token_per_layer=576,  # MLA compressed latent + rope dims
        n_experts=256, topk=8,
    ),
    "moe-1t-a32b": dict(
        label="MoE 1T / 32B active (Kimi-K2-like, MLA)",
        params_total=1040e9, params_active=32e9,
        n_layers=61, d_model=7168, n_q_heads=64, n_kv_heads=64, d_head=128,
        kv_entries_per_token_per_layer=576,
        n_experts=384, topk=8,
    ),
}

# --- GPU presets (approximate public spec-sheet numbers, DENSE tensor FLOPs) --

GPUS = {
    "h100-sxm": dict(label="H100 SXM", tflops_bf16=989, tflops_fp8=1979,
                     hbm_gb=80, hbm_tbps=3.35, interconnect_gbps=900),
    "h200": dict(label="H200", tflops_bf16=989, tflops_fp8=1979,
                 hbm_gb=141, hbm_tbps=4.8, interconnect_gbps=900),
    "b200": dict(label="B200", tflops_bf16=2250, tflops_fp8=4500,
                 hbm_gb=192, hbm_tbps=8.0, interconnect_gbps=1800),
    "mi300x": dict(label="MI300X", tflops_bf16=1307, tflops_fp8=2615,
                   hbm_gb=192, hbm_tbps=5.3, interconnect_gbps=896),
}

DTYPE_BYTES = {"bf16": 2, "fp8": 1}

# What 'any (default)' in the Explorer assumption dropdowns resolves to.
DEFAULT_ASSUMPTIONS = {"arch": "moe-1t-a32b", "gpu": "h100-sxm",
                       "wdtype": "fp8", "kvdtype": "bf16",
                       "tp": 8, "pp": 1, "dp": 1, "mfu": 40, "mbu": 60}


def resolve_assumptions(config: dict | None) -> tuple[str, str, dict]:
    """Explorer config store ({k: value|None}) -> (arch_key, gpu_key, cfg).

    None (the 'any' sentinel) falls back to DEFAULT_ASSUMPTIONS. cfg is the
    dict request_compute/implied_gpu_seconds take (mfu/mbu as fractions).
    """
    c = dict(DEFAULT_ASSUMPTIONS)
    for k, v in (config or {}).items():
        if v is not None:
            if k not in c:
                raise KeyError(f"unknown assumption key {k!r}; allowed: {sorted(c)}")
            c[k] = v
    return c["arch"], c["gpu"], {
        "dtype_weights": c["wdtype"], "dtype_kv": c["kvdtype"],
        "tp": int(c["tp"]), "pp": int(c["pp"]), "dp": int(c["dp"]),
        "mfu": float(c["mfu"]) / 100, "mbu": float(c["mbu"]) / 100,
    }


def _attn_flops(arch: dict, n_new_tokens: float, ctx_start: float) -> float:
    """Attention score+value FLOPs for n new tokens entering at context ctx_start.

    Per token at context c: QK^T + AV ~= 4 * L * Hq * dh * c FLOPs.
    Summed over the n tokens: 4 * L * Hq * dh * n * (ctx_start + n/2).
    """
    a = arch
    return 4.0 * a["n_layers"] * a["n_q_heads"] * a["d_head"] * n_new_tokens * (ctx_start + n_new_tokens / 2.0)


def request_compute(arch: dict, cached_tokens: int, uncached_tokens: int,
                    out_tokens: int, cfg: dict) -> dict:
    """Implied compute / HBM / network for ONE request.

    cfg: dtype_weights ('bf16'|'fp8'), dtype_kv ('bf16'|'fp8'), tp (int >= 1).

    Formulas:
      prefill_flops = 2 * P_active * U            (linear layers)
                    + attn(U tokens from ctx C)   (quadratic term)
      decode_flops  = 2 * P_active * O + attn(O tokens from ctx C+U)
      kv_btl        = kv_entries_per_token_per_layer * kv_bytes   (per token per layer)
      prefill_hbm   = one weight pass (P_active * wbytes, chunked-prefill approx)
                    + read cached KV (C * L * kv_btl) + write new KV (U * L * kv_btl)
      decode_hbm    = O * P_active * wbytes                        (weights re-read per step)
                    + O * (ctx_in + O/2) * L * kv_btl              (KV read per step)
                    + O * L * kv_btl                               (KV write)
      net_tp        = 2 allreduces per layer, ring cost 2*(tp-1)/tp * d_model * abytes
                      per token  ->  4 * L * d_model * abytes * (tp-1)/tp * (U + O)
      net_ep        = dispatch+combine all-to-all: 2 * topk * d_model * abytes
                      per token per layer * (U + O)                (MoE only)
      net_pp        = activations across each stage boundary (forward only):
                      (pp-1) * d_model * abytes * (U + O)
      dp (data parallel) adds no per-request traffic at inference (replication
      only); it is carried in cfg for display/feasibility, not for the math.
    """
    wbytes = DTYPE_BYTES[cfg["dtype_weights"]]
    kvbytes = DTYPE_BYTES[cfg["dtype_kv"]]
    abytes = 2  # activations in bf16
    tp = int(cfg["tp"])
    pp = int(cfg.get("pp", 1))
    if tp < 1 or pp < 1:
        raise ValueError(f"tp and pp must be >= 1, got tp={tp} pp={pp}")
    a = arch
    C, U, O = float(cached_tokens), float(uncached_tokens), float(out_tokens)
    ctx_in = C + U
    kv_btl_bytes = a["kv_entries_per_token_per_layer"] * kvbytes
    L = a["n_layers"]

    prefill_flops = 2.0 * a["params_active"] * U + _attn_flops(a, U, C)
    decode_flops = 2.0 * a["params_active"] * O + _attn_flops(a, O, ctx_in)

    prefill_hbm_bytes = (
        (a["params_active"] * wbytes if U > 0 else 0.0)
        + C * L * kv_btl_bytes
        + U * L * kv_btl_bytes
    )
    decode_hbm_bytes = (
        O * a["params_active"] * wbytes
        + O * (ctx_in + O / 2.0) * L * kv_btl_bytes
        + O * L * kv_btl_bytes
    )

    net_tp_bytes = 4.0 * L * a["d_model"] * abytes * ((tp - 1) / tp) * (U + O)
    net_ep_bytes = (
        2.0 * a["topk"] * L * a["d_model"] * abytes * (U + O) if a["n_experts"] else 0.0
    )
    net_pp_bytes = (pp - 1) * a["d_model"] * abytes * (U + O)

    kv_footprint_bytes = (ctx_in + O) * L * kv_btl_bytes

    return {
        "prefill_flops": prefill_flops,
        "decode_flops": decode_flops,
        "prefill_hbm_bytes": prefill_hbm_bytes,
        "decode_hbm_bytes": decode_hbm_bytes,
        "net_tp_bytes": net_tp_bytes,
        "net_ep_bytes": net_ep_bytes,
        "net_pp_bytes": net_pp_bytes,
        "kv_footprint_bytes": kv_footprint_bytes,
    }


def conversation_compute(records: list[dict], arch: dict, cfg: dict) -> dict:
    """Aggregate request_compute over a conversation's records.

    Returns {'totals': {...sums..., 'peak_kv_bytes', 'n_requests', token sums},
             'per_request': [{'uid','start_s','end_s','role','model','in_tokens',
                              'out_tokens','flops','hbm_bytes'}]}.
    """
    if not records:
        raise ValueError("conversation_compute on empty records")
    totals = {
        "prefill_flops": 0.0, "decode_flops": 0.0,
        "prefill_hbm_bytes": 0.0, "decode_hbm_bytes": 0.0,
        "net_tp_bytes": 0.0, "net_ep_bytes": 0.0, "net_pp_bytes": 0.0,
        "peak_kv_bytes": 0.0,
        "n_requests": len(records),
        "in_tokens": 0, "cached_tokens": 0, "uncached_tokens": 0, "out_tokens": 0,
    }
    per_request = []
    for r in records:
        c = request_compute(arch, r["cached_tokens"], r["uncached_tokens"], r["out_tokens"], cfg)
        for k in ("prefill_flops", "decode_flops", "prefill_hbm_bytes",
                  "decode_hbm_bytes", "net_tp_bytes", "net_ep_bytes", "net_pp_bytes"):
            totals[k] += c[k]
        totals["peak_kv_bytes"] = max(totals["peak_kv_bytes"], c["kv_footprint_bytes"])
        for k in ("in_tokens", "cached_tokens", "uncached_tokens", "out_tokens"):
            totals[k] += r[k]
        per_request.append({
            "uid": r["uid"], "start_s": r["start_s"], "end_s": r["end_s"],
            "role": r["role"], "model": r["model"],
            "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"],
            "cached_tokens": r["cached_tokens"],
            "uncached_tokens": r["uncached_tokens"],
            "flops": c["prefill_flops"] + c["decode_flops"],
            "hbm_bytes": c["prefill_hbm_bytes"] + c["decode_hbm_bytes"],
        })
    return {"totals": totals, "per_request": per_request}


def implied_gpu_seconds(totals: dict, gpu: dict, cfg: dict) -> dict:
    """Single-GPU-equivalent busy time implied by the totals.

    compute_s = flops / (peak_dense_tflops[dtype] * 1e12 * mfu)
    memory_s  = hbm_bytes / (hbm_tbps * 1e12 * mbu)
    bound     = whichever is larger (per-phase: prefill compare, decode compare).
    """
    mfu = float(cfg["mfu"])
    mbu = float(cfg["mbu"])
    if not (0 < mfu <= 1 and 0 < mbu <= 1):
        raise ValueError(f"mfu/mbu must be in (0, 1], got mfu={mfu} mbu={mbu}")
    peak_flops_per_s = gpu[f"tflops_{cfg['dtype_weights']}"] * 1e12 * mfu
    hbm_bytes_per_s = gpu["hbm_tbps"] * 1e12 * mbu

    prefill_compute_s = totals["prefill_flops"] / peak_flops_per_s
    prefill_memory_s = totals["prefill_hbm_bytes"] / hbm_bytes_per_s
    decode_compute_s = totals["decode_flops"] / peak_flops_per_s
    decode_memory_s = totals["decode_hbm_bytes"] / hbm_bytes_per_s

    prefill_s = max(prefill_compute_s, prefill_memory_s)
    decode_s = max(decode_compute_s, decode_memory_s)
    return {
        "prefill_compute_s": prefill_compute_s,
        "prefill_memory_s": prefill_memory_s,
        "decode_compute_s": decode_compute_s,
        "decode_memory_s": decode_memory_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": prefill_s + decode_s,
        "prefill_bound": "compute" if prefill_compute_s >= prefill_memory_s else "memory",
        "decode_bound": "compute" if decode_compute_s >= decode_memory_s else "memory",
    }
