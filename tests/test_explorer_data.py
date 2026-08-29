"""Tests for Explorer-tab pure helpers: table rows, curves, selection filter."""
import unittest

from modules.arch import resolve_assumptions
from modules.explorer_data import (apply_selection, augment_rows_with_compute,
                                   build_conversation_table, conversation_curves)


def _rec(conv_id, role="main", turn_index=0, in_t=100, out=10,
         start=0.0, end=1.0, model="m1"):
    return {"uid": f"{conv_id}#{role}{turn_index}", "conv_id": conv_id,
            "role": role, "agent_id": None, "depth": 0 if role == "main" else 1,
            "model": model, "in_tokens": in_t, "cached_tokens": in_t - 10,
            "uncached_tokens": 10, "out_tokens": out,
            "start_s": start, "end_s": end, "turn_index": turn_index}


class TestConversationTable(unittest.TestCase):
    def setUp(self):
        self.pool = [
            _rec("cA", turn_index=0, in_t=3000, start=0, end=5),
            _rec("cA", turn_index=1, in_t=50000, start=10, end=20),
            _rec("cA", role="subagent", in_t=8000, start=3600, end=7200),
            _rec("cB", turn_index=0, in_t=1000, start=0, end=30, model="m2"),
        ]
        self.index = ["cA", "cB"]

    def test_rows_shape_and_metrics(self):
        rows, skipped = build_conversation_table(self.pool, self.index, None)
        self.assertEqual(skipped, 0)
        self.assertEqual([r["ordinal"] for r in rows], [1, 2])
        a = rows[0]
        self.assertEqual(a["id"], "cA")
        self.assertEqual(a["model"], "m1")         # first main turn's model
        self.assertEqual(a["n_turns"], 2)          # main turns only
        self.assertEqual(a["final_ctx"], 50000)    # last main turn's context
        self.assertEqual(a["max_ctx"], 50000)      # >= final_ctx by definition
        self.assertEqual(a["isl0"], 3000)          # first main turn's context
        self.assertEqual(a["avg_out"], 10.0)       # mean decode out of main turns
        self.assertEqual(a["max_out"], 10)
        self.assertEqual(a["duration_h"], 2.0)     # subagent end at 7200s counts

    def test_max_ctx_can_exceed_final_ctx(self):
        pool = [_rec("cC", turn_index=0, in_t=1000),
                _rec("cC", turn_index=1, in_t=900000),   # pre-compaction peak
                _rec("cC", turn_index=2, in_t=200000)]   # compacted final
        rows, _ = build_conversation_table(pool, ["cC"], None)
        self.assertEqual(rows[0]["max_ctx"], 900000)
        self.assertEqual(rows[0]["final_ctx"], 200000)
        self.assertGreater(rows[0]["max_ctx"], rows[0]["final_ctx"])

    def test_augment_adds_assumptions_and_gpu_counts(self):
        rows, _ = build_conversation_table(self.pool, self.index, None)
        arch_key, gpu_key, cfg = resolve_assumptions(None)
        out = augment_rows_with_compute(rows, self.pool, arch_key, gpu_key, cfg)
        self.assertEqual(len(out), len(rows))
        self.assertNotIn("gpu", rows[0])  # copy-on-write: input untouched
        a = out[0]
        self.assertEqual(a["gpu"], "H100 SXM")
        self.assertEqual(a["wq"], "fp8")
        self.assertEqual(a["kvq"], "bf16")
        self.assertEqual((a["tp"], a["pp"], a["dp"]), (8, 1, 1))
        self.assertGreater(a["prefill_gpus"], 0)
        self.assertGreater(a["decode_gpus"], 0)

    def test_augment_zero_wallclock_gives_none_not_zero(self):
        pool = [_rec("cZ", start=5.0, end=5.0)]
        rows, _ = build_conversation_table(pool, ["cZ"], None)
        arch_key, gpu_key, cfg = resolve_assumptions(None)
        out = augment_rows_with_compute(rows, pool, arch_key, gpu_key, cfg)
        self.assertIsNone(out[0]["prefill_gpus"])  # never fabricate a 0
        self.assertIsNone(out[0]["decode_gpus"])

    def test_augment_missing_conv_raises(self):
        rows, _ = build_conversation_table(self.pool, self.index, None)
        arch_key, gpu_key, cfg = resolve_assumptions(None)
        with self.assertRaises(ValueError):
            augment_rows_with_compute(rows, [r for r in self.pool
                                             if r["conv_id"] != "cB"],
                                      arch_key, gpu_key, cfg)

    def test_model_filter_any_of(self):
        rows, _ = build_conversation_table(self.pool, self.index, ["m2"])
        self.assertEqual([r["id"] for r in rows], ["cB"])

    def test_conv_missing_from_index_raises(self):
        with self.assertRaises(ValueError):
            build_conversation_table(self.pool, ["cA"], None)  # cB not in index

    def test_no_main_turns_skipped_not_fabricated(self):
        pool = [_rec("cC", role="subagent")]
        rows, skipped = build_conversation_table(pool, ["cC"], None)
        self.assertEqual(rows, [])
        self.assertEqual(skipped, 1)


class TestCurves(unittest.TestCase):
    def test_curve_ordering_and_axes(self):
        pool = [
            _rec("cA", turn_index=1, in_t=200, start=10),
            _rec("cA", turn_index=0, in_t=100, start=0),
            _rec("cA", role="subagent", in_t=999),  # excluded from the curve
        ]
        curves = conversation_curves(pool)
        xs, ys = curves["cA"]
        self.assertEqual(xs, [1, 2])
        self.assertEqual(ys, [100, 200])  # sorted by turn_index, not input order

    def test_conv_ids_limit(self):
        pool = [_rec("cA"), _rec("cB")]
        curves = conversation_curves(pool, conv_ids={"cB"})
        self.assertEqual(set(curves), {"cB"})


class TestApplySelection(unittest.TestCase):
    def setUp(self):
        self.pool = [_rec("cA"), _rec("cA", turn_index=1), _rec("cB")]

    def test_applies_on_matching_slug(self):
        out, gates = apply_selection(self.pool, {"slug": "ds", "conv_ids": ["cA"]}, "ds")
        self.assertEqual(len(out), 2)
        self.assertEqual(gates["n_sel_convs"], 1)
        self.assertEqual(gates["n_after_selection"], 2)

    def test_ignored_on_other_dataset(self):
        out, gates = apply_selection(self.pool, {"slug": "other", "conv_ids": ["cA"]}, "ds")
        self.assertEqual(len(out), 3)  # not silently zeroed
        self.assertEqual(gates, {})

    def test_no_selection_passthrough(self):
        for sel in (None, {}, {"slug": "ds", "conv_ids": []}):
            out, gates = apply_selection(self.pool, sel, "ds")
            self.assertEqual(len(out), 3)
            self.assertEqual(gates, {})


class TestSyncUpdates(unittest.TestCase):
    """Shared-dataset propagation: one value, N synced widgets."""

    def setUp(self):
        from modules.explorer_data import sync_updates
        self.sync = sync_updates

    def test_propagates_trigger_value_to_stale_widgets(self):
        self.assertEqual(self.sync(["b", "a", "a"], 0), ("b", [1, 2]))
        self.assertEqual(self.sync(["a", "b", "a"], 1), ("b", [0, 2]))

    def test_converged_returns_none_stopping_the_echo(self):
        self.assertIsNone(self.sync(["a", "a", "a"], 2))
        self.assertIsNone(self.sync([None, None, None], 0))

    def test_clearing_propagates_the_none_value(self):
        # a cleared dropdown clears the others: None is a real value here,
        # "no change" is signaled by index absence
        self.assertEqual(self.sync([None, "a", "a"], 0), (None, [1, 2]))

    def test_partial_agreement_updates_only_stale(self):
        self.assertEqual(self.sync(["b", "b", "a"], 0), ("b", [2]))

    def test_bad_index_raises(self):
        with self.assertRaises(IndexError):
            self.sync(["a"], 3)


if __name__ == "__main__":
    unittest.main()
