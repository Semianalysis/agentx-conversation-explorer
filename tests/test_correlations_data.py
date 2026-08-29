"""Tests for the correlations conditional-subset helper."""
import unittest

from modules.correlations_data import conditional_subset


def _rec(in_t, unc, out):
    return {"in_tokens": in_t, "cached_tokens": in_t - unc,
            "uncached_tokens": unc, "out_tokens": out,
            "model": "m", "role": "main"}


class TestConditionalSubset(unittest.TestCase):
    def setUp(self):
        self.pool = [_rec(100, 10, 5), _rec(1000, 100, 50), _rec(10000, 1000, 500)]

    def test_range_inclusive(self):
        subset, gates = conditional_subset(self.pool, "context", 100, 1000)
        self.assertEqual(len(subset), 2)
        self.assertEqual(gates, {"n_pool": 3, "n_in_range": 2})

    def test_unbounded_sides(self):
        subset, _ = conditional_subset(self.pool, "output", None, 50)
        self.assertEqual(len(subset), 2)
        subset, _ = conditional_subset(self.pool, "new_input", 100, None)
        self.assertEqual(len(subset), 2)
        subset, _ = conditional_subset(self.pool, "context", None, None)
        self.assertEqual(len(subset), 3)

    def test_unknown_dim_raises(self):
        with self.assertRaises(KeyError):
            conditional_subset(self.pool, "decode", 0, 1)

    def test_inverted_range_raises(self):
        with self.assertRaises(ValueError):
            conditional_subset(self.pool, "context", 1000, 100)


if __name__ == "__main__":
    unittest.main()
