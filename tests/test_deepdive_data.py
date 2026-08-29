"""Tests for the deep-dive multi-conversation aggregation."""
import unittest

from modules.arch import resolve_assumptions, ARCHITECTURES
from modules.deepdive_data import aggregate_selection


def _rec(conv_id, role="main", turn_index=0, in_t=1000, out=50,
         start=0.0, end=10.0, model="m1"):
    return {"uid": f"{conv_id}#{role}{turn_index}", "conv_id": conv_id,
            "role": role, "agent_id": None, "depth": 0 if role == "main" else 1,
            "model": model, "in_tokens": in_t, "cached_tokens": in_t - 100,
            "uncached_tokens": 100, "out_tokens": out,
            "start_s": start, "end_s": end, "turn_index": turn_index}


class TestAggregateSelection(unittest.TestCase):
    def setUp(self):
        self.pool = [
            _rec("cA", turn_index=0, start=0, end=10),
            _rec("cA", turn_index=1, in_t=200000, start=20, end=3600),  # big ctx
            _rec("cB", turn_index=0, start=0, end=7200),
        ]
        self.ordinal = {"cA": 1, "cB": 2}
        arch_key, _, self.cfg = resolve_assumptions(None)
        self.arch = ARCHITECTURES[arch_key]

    def test_totals_sum_and_wall_sum(self):
        agg = aggregate_selection(self.pool, ["cA", "cB"], self.arch, self.cfg,
                                  self.ordinal)
        self.assertEqual(agg["n_convs"], 2)
        self.assertEqual(agg["totals"]["n_requests"], 3)
        self.assertEqual(agg["wall_s_sum"], 3600 + 7200)
        a_only = aggregate_selection(self.pool, ["cA"], self.arch, self.cfg,
                                     self.ordinal)
        b_only = aggregate_selection(self.pool, ["cB"], self.arch, self.cfg,
                                     self.ordinal)
        self.assertAlmostEqual(
            agg["totals"]["prefill_flops"],
            a_only["totals"]["prefill_flops"] + b_only["totals"]["prefill_flops"])

    def test_peak_kv_is_max_not_sum(self):
        agg = aggregate_selection(self.pool, ["cA", "cB"], self.arch, self.cfg,
                                  self.ordinal)
        a_only = aggregate_selection(self.pool, ["cA"], self.arch, self.cfg,
                                     self.ordinal)
        self.assertEqual(agg["totals"]["peak_kv_bytes"],
                         a_only["totals"]["peak_kv_bytes"])  # cA has the big ctx

    def test_per_request_carries_ordinal_and_order(self):
        agg = aggregate_selection(self.pool, ["cB", "cA"], self.arch, self.cfg,
                                  self.ordinal)
        ords = [p["ord"] for p in agg["per_request"]]
        self.assertEqual(ords, sorted(ords))  # ordered by ordinal despite input order
        self.assertEqual({p["cid"] for p in agg["per_request"]}, {"cA", "cB"})

    def test_empty_selection_raises(self):
        with self.assertRaises(ValueError):
            aggregate_selection(self.pool, [], self.arch, self.cfg, self.ordinal)

    def test_missing_conversation_raises(self):
        with self.assertRaises(ValueError):
            aggregate_selection(self.pool, ["cZ"], self.arch, self.cfg, self.ordinal)


if __name__ == "__main__":
    unittest.main()
