"""Tests for the measure registry and the enriched per-request rows."""
import unittest

from modules.arch import ARCHITECTURES, DTYPE_BYTES, resolve_assumptions
from modules.deepdive_data import aggregate_selection
from modules.measures import (ALL_MEASURES, DEFAULT_AXES, X_MEASURES,
                              Y_MEASURES, measure_series, shared_axis_range)


def _rec(conv_id, turn_index=0, in_t=1000, unc=100, out=50, start=0.0, end=10.0):
    return {"uid": f"{conv_id}#{turn_index}", "conv_id": conv_id, "role": "main",
            "agent_id": None, "depth": 0, "model": "m",
            "in_tokens": in_t, "cached_tokens": in_t - unc,
            "uncached_tokens": unc, "out_tokens": out,
            "start_s": start, "end_s": end, "turn_index": turn_index}


class TestMeasureRegistry(unittest.TestCase):
    def setUp(self):
        pool = [
            _rec("cA", 0, start=0.0, end=0.5),      # starts before 1 s
            _rec("cA", 1, start=120.0, end=130.0),
            _rec("cB", 0, start=0.2, end=3.0, unc=0, out=0),  # zero-valued measures
        ]
        arch_key, _, self.cfg = resolve_assumptions(None)
        self.arch = ARCHITECTURES[arch_key]
        self.agg = aggregate_selection(pool, ["cA", "cB"], self.arch, self.cfg,
                                       {"cA": 1, "cB": 2})

    def test_registry_shape(self):
        self.assertEqual(set(DEFAULT_AXES), {"x", "y1", "y2", "y3"})
        self.assertIn(DEFAULT_AXES["x"], X_MEASURES)
        for a in ("y1", "y2", "y3"):
            self.assertIn(DEFAULT_AXES[a], Y_MEASURES)
        for key, m in ALL_MEASURES.items():
            self.assertIn(m["scale"], ("linear", "log"), key)

    def test_ordinals_linear_magnitudes_log(self):
        self.assertEqual(X_MEASURES["conv_number"]["scale"], "linear")
        self.assertEqual(X_MEASURES["turn_number"]["scale"], "linear")
        self.assertEqual(X_MEASURES["cumulative_time"]["scale"], "log")
        for key, m in Y_MEASURES.items():
            self.assertEqual(m["scale"], "log", key)

    def test_log_measures_clamp_to_one_not_dropped(self):
        t = measure_series(self.agg["per_request"], "cumulative_time")
        self.assertEqual(len(t), 3)          # sub-1s starts kept, not dropped
        self.assertTrue(all(v >= 1.0 for v in t))
        unc = measure_series(self.agg["per_request"], "new_input_tokens")
        self.assertTrue(all(v >= 1.0 for v in unc))  # the zero clamps to 1
        out = measure_series(self.agg["per_request"], "new_output_tokens")
        self.assertTrue(all(v >= 1.0 for v in out))

    def test_seq_and_cum_flops_enrichment(self):
        rows = [p for p in self.agg["per_request"] if p["cid"] == "cA"]
        self.assertEqual([p["seq"] for p in rows], [1, 2])
        self.assertAlmostEqual(rows[1]["cum_flops"],
                               rows[0]["flops"] + rows[1]["flops"])
        self.assertGreater(rows[1]["cum_flops"], rows[0]["cum_flops"])

    def test_kv_bytes_enrichment(self):
        p = self.agg["per_request"][0]
        kv_per_tok = (self.arch["n_layers"]
                      * self.arch["kv_entries_per_token_per_layer"]
                      * DTYPE_BYTES[self.cfg["dtype_kv"]])
        self.assertEqual(p["kv_bytes"], p["in_tokens"] * kv_per_tok)
        self.assertEqual(measure_series([p], "context_kv_bytes")[0], p["kv_bytes"])

    def test_unknown_measure_raises(self):
        with self.assertRaises(KeyError):
            measure_series(self.agg["per_request"], "contxt_tokens")

    def test_cum_out_enrichment(self):
        rows = [p for p in self.agg["per_request"] if p["cid"] == "cA"]
        self.assertEqual([p["cum_out"] for p in rows], [50, 100])
        self.assertEqual(measure_series(rows, "cumulative_output_tokens"), [50, 100])


class TestSharedAxisRange(unittest.TestCase):
    def test_log_range_is_log10_and_covers_values(self):
        rng = shared_axis_range([1.0, 1000.0], "log")
        self.assertLess(rng[0], 0.0)          # log10(1) minus pad
        self.assertGreater(rng[1], 3.0)       # log10(1000) plus pad
        # pad is 2% of the 3-decade span per side
        self.assertAlmostEqual(rng[0], -0.06)
        self.assertAlmostEqual(rng[1], 3.06)

    def test_log_single_value_gets_minimum_pad(self):
        rng = shared_axis_range([100.0], "log")
        self.assertAlmostEqual(rng[0], 2.0 - 0.05)
        self.assertAlmostEqual(rng[1], 2.0 + 0.05)

    def test_linear_range_covers_ordinals(self):
        rng = shared_axis_range([1, 2, 3], "linear")
        self.assertLessEqual(rng[0], 1 - 0.5 + 1e-9)
        self.assertGreaterEqual(rng[1], 3 + 0.5 - 1e-9)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            shared_axis_range([], "log")

    def test_nonpositive_log_raises(self):
        with self.assertRaises(ValueError):
            shared_axis_range([0.0, 10.0], "log")

    def test_same_range_for_every_measure_over_one_pool(self):
        # The lock property: the range depends only on the x series, so any
        # chart built over the same per-request rows gets the identical window.
        xs = [1.0, 55.0, 778137.0]
        self.assertEqual(shared_axis_range(xs, "log"),
                         shared_axis_range(list(reversed(xs)), "log"))


class TestGroupedSeries(unittest.TestCase):
    def setUp(self):
        from modules.deepdive_data import grouped_series
        self.grouped_series = grouped_series
        pool = [
            _rec("cA", 0, in_t=1000, out=100, start=0.0, end=1.0),
            _rec("cA", 1, in_t=3000, out=300, start=10.0, end=11.0),  # cB has ended
            _rec("cB", 0, in_t=2000, out=200, start=0.0, end=1.0),
        ]
        arch_key, _, cfg = resolve_assumptions(None)
        self.agg = aggregate_selection(pool, ["cA", "cB"], ARCHITECTURES[arch_key],
                                       cfg, {"cA": 1, "cB": 2})

    def test_turn_number_mean_and_dropout(self):
        g = self.grouped_series(self.agg["per_request"], "turn_number",
                                "context_tokens")
        self.assertEqual(g["xs"], [1, 2])
        self.assertEqual(g["n_alive"], [2, 1])       # cB dropped out at turn 2
        self.assertAlmostEqual(g["mean"][0], 1500.0)  # (1000+2000)/2
        self.assertAlmostEqual(g["mean"][1], 3000.0)  # only cA — no padding
        self.assertEqual((g["lo"][0], g["hi"][0]), (1000.0, 2000.0))
        self.assertEqual((g["lo"][1], g["hi"][1]), (3000.0, 3000.0))

    def test_envelope_brackets_mean(self):
        g = self.grouped_series(self.agg["per_request"], "cumulative_time",
                                "new_output_tokens", n_bins=10)
        for lo, m, hi in zip(g["lo"], g["mean"], g["hi"]):
            self.assertLessEqual(lo, m)
            self.assertLessEqual(m, hi)

    def test_conv_number_grouping_raises(self):
        with self.assertRaises(ValueError):
            self.grouped_series(self.agg["per_request"], "conv_number",
                                "context_tokens")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            self.grouped_series([], "turn_number", "context_tokens")


if __name__ == "__main__":
    unittest.main()
