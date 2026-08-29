"""Tests for the implied-compute math: hand-checked small cases + monotonicity."""
import unittest

from modules.arch import (ARCHITECTURES, DEFAULT_ASSUMPTIONS, GPUS,
                          conversation_compute, implied_gpu_seconds,
                          request_compute, resolve_assumptions)

CFG = {"dtype_weights": "bf16", "dtype_kv": "bf16", "tp": 8, "mfu": 0.4, "mbu": 0.6}

TINY = dict(  # hand-checkable architecture
    label="tiny", params_total=1e6, params_active=1e6,
    n_layers=2, d_model=64, n_q_heads=4, n_kv_heads=2, d_head=16,
    kv_entries_per_token_per_layer=2 * 2 * 16,
    n_experts=0, topk=0,
)


def _rec(cached, uncached, out, start=0.0, end=1.0, uid="u", role="main"):
    return {"uid": uid, "conv_id": "c", "role": role, "agent_id": None, "depth": 0,
            "model": "m", "in_tokens": cached + uncached, "cached_tokens": cached,
            "uncached_tokens": uncached, "out_tokens": out,
            "start_s": start, "end_s": end, "turn_index": 0}


class TestRequestCompute(unittest.TestCase):
    def test_linear_flops_hand_calc(self):
        # 100 uncached tokens from empty context, no decode:
        # linear = 2 * 1e6 * 100 = 2e8; attn = 4*2*4*16*100*(0+50) = 2.56e6
        c = request_compute(TINY, 0, 100, 0, CFG)
        self.assertAlmostEqual(c["prefill_flops"], 2e8 + 4 * 2 * 4 * 16 * 100 * 50)
        self.assertEqual(c["decode_flops"], 0.0)

    def test_decode_flops_grow_with_context(self):
        small = request_compute(TINY, 100, 10, 50, CFG)
        big = request_compute(TINY, 100000, 10, 50, CFG)
        self.assertGreater(big["decode_flops"], small["decode_flops"])
        self.assertGreater(big["decode_hbm_bytes"], small["decode_hbm_bytes"])

    def test_kv_bytes_hand_calc(self):
        # kv per token per layer = 64 entries * 2 bytes = 128 B; L=2 -> 256 B/token
        c = request_compute(TINY, 0, 100, 0, CFG)
        # prefill hbm = weights (1e6*2) + write 100 tokens * 256
        self.assertAlmostEqual(c["prefill_hbm_bytes"], 2e6 + 100 * 256)

    def test_cached_prefill_skips_weight_pass(self):
        c = request_compute(TINY, 1000, 0, 0, CFG)  # fully cached, nothing new
        self.assertEqual(c["prefill_flops"], 0.0)
        self.assertAlmostEqual(c["prefill_hbm_bytes"], 1000 * 256)  # KV re-read only

    def test_tp1_has_zero_tp_traffic(self):
        cfg = dict(CFG, tp=1)
        c = request_compute(TINY, 0, 100, 10, cfg)
        self.assertEqual(c["net_tp_bytes"], 0.0)

    def test_moe_has_ep_traffic_dense_none(self):
        dense = request_compute(ARCHITECTURES["dense-70b"], 0, 100, 10, CFG)
        moe = request_compute(ARCHITECTURES["moe-671b-a37b"], 0, 100, 10, CFG)
        self.assertEqual(dense["net_ep_bytes"], 0.0)
        self.assertGreater(moe["net_ep_bytes"], 0.0)

    def test_bad_tp_raises(self):
        with self.assertRaises(ValueError):
            request_compute(TINY, 0, 1, 1, dict(CFG, tp=0))

    def test_pp_traffic(self):
        pp1 = request_compute(TINY, 0, 100, 10, dict(CFG, pp=1))
        pp4 = request_compute(TINY, 0, 100, 10, dict(CFG, pp=4))
        self.assertEqual(pp1["net_pp_bytes"], 0.0)
        # (pp-1) * d_model * abytes * (U+O) = 3 * 64 * 2 * 110
        self.assertAlmostEqual(pp4["net_pp_bytes"], 3 * 64 * 2 * 110)


class TestResolveAssumptions(unittest.TestCase):
    def test_all_any_gives_defaults(self):
        arch_key, gpu_key, cfg = resolve_assumptions(
            {k: None for k in DEFAULT_ASSUMPTIONS})
        self.assertEqual(arch_key, DEFAULT_ASSUMPTIONS["arch"])
        self.assertEqual(gpu_key, DEFAULT_ASSUMPTIONS["gpu"])
        self.assertEqual(cfg["tp"], DEFAULT_ASSUMPTIONS["tp"])
        self.assertEqual(cfg["pp"], 1)
        self.assertAlmostEqual(cfg["mfu"], DEFAULT_ASSUMPTIONS["mfu"] / 100)

    def test_override_and_empty(self):
        _, gpu_key, cfg = resolve_assumptions({"gpu": "b200", "tp": 16})
        self.assertEqual(gpu_key, "b200")
        self.assertEqual(cfg["tp"], 16)
        arch_key, _, _ = resolve_assumptions(None)
        self.assertEqual(arch_key, DEFAULT_ASSUMPTIONS["arch"])

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            resolve_assumptions({"gup": "b200"})  # typo must be loud


class TestConversationCompute(unittest.TestCase):
    def test_totals_sum_and_peak(self):
        recs = [_rec(0, 100, 10, uid="a"), _rec(100, 50, 20, uid="b")]
        result = conversation_compute(recs, TINY, CFG)
        t = result["totals"]
        self.assertEqual(t["n_requests"], 2)
        self.assertEqual(t["uncached_tokens"], 150)
        self.assertEqual(t["out_tokens"], 30)
        # peak KV = max over requests of (in + out) * L * kv_btl = 170 * 256
        self.assertAlmostEqual(t["peak_kv_bytes"], 170 * 256)
        self.assertEqual(len(result["per_request"]), 2)

    def test_empty_records_raise(self):
        with self.assertRaises(ValueError):
            conversation_compute([], TINY, CFG)


class TestImpliedGpuSeconds(unittest.TestCase):
    def test_hand_calc(self):
        totals = {"prefill_flops": 989e12 * 0.4, "decode_flops": 0.0,
                  "prefill_hbm_bytes": 0.0, "decode_hbm_bytes": 0.0}
        out = implied_gpu_seconds(totals, GPUS["h100-sxm"], CFG)
        self.assertAlmostEqual(out["prefill_compute_s"], 1.0)  # exactly 1s of H100 @40% MFU
        self.assertEqual(out["prefill_bound"], "compute")

    def test_bad_mfu_raises(self):
        with self.assertRaises(ValueError):
            implied_gpu_seconds({"prefill_flops": 1, "decode_flops": 1,
                                 "prefill_hbm_bytes": 1, "decode_hbm_bytes": 1},
                                GPUS["h100-sxm"], dict(CFG, mfu=0))

    def test_all_presets_complete(self):
        recs = [_rec(1000, 200, 50)]
        for key, arch in ARCHITECTURES.items():
            result = conversation_compute(recs, arch, CFG)
            for gkey, gpu in GPUS.items():
                out = implied_gpu_seconds(result["totals"], gpu, CFG)
                self.assertGreater(out["total_s"], 0, f"{key} x {gkey}")


if __name__ == "__main__":
    unittest.main()
