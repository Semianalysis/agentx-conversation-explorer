"""Tests for Turns-tab pure helpers: filters, binning, stats."""
import unittest

from modules.binning import (bin_counts, dimension_values, filter_records,
                                make_bins, value_stats)


def _rec(model="m1", role="main", in_t=100, unc=40, out=10):
    return {"model": model, "role": role, "in_tokens": in_t,
            "cached_tokens": in_t - unc, "uncached_tokens": unc, "out_tokens": out}


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.pool = [
            _rec("m1", "main"), _rec("m1", "subagent"),
            _rec("m2", "main"), _rec("m2", "subagent"),
        ]

    def test_no_filters_passthrough(self):
        out, gates = filter_records(self.pool, {})
        self.assertEqual(len(out), 4)
        self.assertEqual(gates["n_pool"], 4)

    def test_model_and_role(self):
        out, gates = filter_records(self.pool, {"models": ["m1"], "roles": ["main"]})
        self.assertEqual(len(out), 1)
        self.assertEqual(gates, {"n_pool": 4, "n_model": 2, "n_role": 1})

    def test_unknown_filter_key_raises(self):
        with self.assertRaises(KeyError):
            filter_records(self.pool, {"modles": ["m1"]})  # typo must be loud

    def test_unknown_dimension_raises(self):
        with self.assertRaises(KeyError):
            dimension_values(self.pool, "contxt")


class TestBinning(unittest.TestCase):
    def test_empty_values_raise(self):
        with self.assertRaises(ValueError):
            make_bins([], 10, log_x=True)
        with self.assertRaises(ValueError):
            value_stats([])

    def test_linear_bins_cover_all_values(self):
        values = [0, 5, 10, 99]
        edges = make_bins(values, 10, log_x=False)
        counts = bin_counts(values, edges, log_x=False)
        self.assertEqual(sum(counts), len(values))
        self.assertEqual(len(edges), 11)

    def test_log_bins_clamp_zero_to_one(self):
        values = [0, 1, 10, 1000]
        edges = make_bins(values, 6, log_x=True)
        counts = bin_counts(values, edges, log_x=True)
        self.assertEqual(sum(counts), len(values))  # the 0 landed in bin 0
        self.assertGreaterEqual(counts[0], 2)  # 0 (clamped) and 1 share bin 0

    def test_log_edges_geometric(self):
        edges = make_bins([1, 10000], 4, log_x=True)
        ratios = [edges[i + 1] / edges[i] for i in range(3)]
        for r in ratios[1:]:
            self.assertAlmostEqual(r, ratios[0], places=6)

    def test_out_of_range_value_raises(self):
        edges = make_bins([1, 100], 5, log_x=False)
        with self.assertRaises(ValueError):
            bin_counts([1000], edges, log_x=False)

    def test_constant_values_bin(self):
        values = [7, 7, 7]
        for log_x in (True, False):
            edges = make_bins(values, 5, log_x=log_x)
            self.assertEqual(sum(bin_counts(values, edges, log_x)), 3)

    def test_value_stats(self):
        s = value_stats([1, 2, 3, 4, 100])
        self.assertEqual(s["count"], 5)
        self.assertEqual(s["median"], 3)
        self.assertEqual(s["max"], 100)
        self.assertAlmostEqual(s["mean"], 22.0)


if __name__ == "__main__":
    unittest.main()
