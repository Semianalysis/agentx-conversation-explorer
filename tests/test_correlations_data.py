"""Tests for the Correlations selection-store state machine and bin matching."""
import unittest

from modules.correlations_data import (MAX_SELECTIONS, add_bin_range,
                                       add_selection, bin_runs, bins_subset,
                                       delete_selection, initial_store,
                                       selection_color, set_applied, set_live,
                                       toggle_bin)
from modules.figures import selection_to_bins
from modules.turns_data import bin_index, make_bins


def _rec(context=1000, new_input=100, output=50):
    return {"in_tokens": context, "uncached_tokens": new_input,
            "out_tokens": output}


class TestSelectionStore(unittest.TestCase):
    def test_initial_shape(self):
        s = initial_store()
        self.assertEqual(len(s["selections"]), 1)
        self.assertEqual(s["live"], s["selections"][0]["sid"])
        self.assertFalse(s["applied"])
        self.assertIsNone(s["selections"][0]["dim"])

    def test_toggle_anchors_then_toggles(self):
        s = toggle_bin(initial_store(), 0, "context", 5)
        self.assertEqual(s["selections"][0]["dim"], "context")
        self.assertEqual(s["selections"][0]["bins"], [5])
        s = toggle_bin(s, 0, "context", 2)
        self.assertEqual(s["selections"][0]["bins"], [2, 5])  # kept sorted
        s = toggle_bin(s, 0, "context", 5)
        self.assertEqual(s["selections"][0]["bins"], [2])

    def test_removing_last_bin_unanchors(self):
        s = toggle_bin(initial_store(), 0, "context", 5)
        s = toggle_bin(s, 0, "context", 5)
        self.assertIsNone(s["selections"][0]["dim"])
        # re-anchorable to a different dim afterwards
        s = toggle_bin(s, 0, "output", 3)
        self.assertEqual(s["selections"][0]["dim"], "output")

    def test_click_on_other_dim_raises_with_hint(self):
        s = toggle_bin(initial_store(), 0, "context", 5)
        with self.assertRaises(ValueError):
            toggle_bin(s, 0, "output", 1)

    def test_unknown_dim_or_sid_raise(self):
        with self.assertRaises(KeyError):
            toggle_bin(initial_store(), 0, "contxt", 1)
        with self.assertRaises(KeyError):
            toggle_bin(initial_store(), 99, "context", 1)

    def test_add_bin_range_unions(self):
        s = toggle_bin(initial_store(), 0, "context", 9)
        s = add_bin_range(s, 0, "context", 2, 4)
        self.assertEqual(s["selections"][0]["bins"], [2, 3, 4, 9])

    def test_add_selection_becomes_live_and_caps(self):
        s = initial_store()
        s = add_selection(s)
        self.assertEqual(len(s["selections"]), 2)
        self.assertEqual(s["live"], s["selections"][-1]["sid"])
        while len(s["selections"]) < MAX_SELECTIONS:
            s = add_selection(s)
        with self.assertRaises(ValueError):
            add_selection(s)

    def test_delete_keeps_one_section_and_fixes_live(self):
        s = add_selection(initial_store())          # sids [0, 1], live 1
        s = delete_selection(s, 1)
        self.assertEqual([x["sid"] for x in s["selections"]], [0])
        self.assertEqual(s["live"], 0)
        s = delete_selection(s, 0)                  # last one -> fresh empty
        self.assertEqual(len(s["selections"]), 1)
        self.assertIsNone(s["selections"][0]["dim"])
        self.assertEqual(s["live"], s["selections"][0]["sid"])

    def test_set_live_and_applied(self):
        s = add_selection(initial_store())
        s = set_live(s, 0)
        self.assertEqual(s["live"], 0)
        with self.assertRaises(KeyError):
            set_live(s, 42)
        s = set_applied(s, True)
        self.assertTrue(s["applied"])

    def test_mutations_are_copy_on_write(self):
        before = initial_store()
        toggle_bin(before, 0, "context", 1)
        self.assertEqual(before["selections"][0]["bins"], [])

    def test_selection_colors_stable_and_distinct(self):
        self.assertEqual(selection_color(3), selection_color(3))
        self.assertNotEqual(selection_color(0), selection_color(1))


class TestBinsSubset(unittest.TestCase):
    def setUp(self):
        # values 1..1000 over 10 log bins; context field drives membership
        self.records = [_rec(context=v) for v in (1, 2, 30, 500, 999, 0)]
        self.values = [r["in_tokens"] for r in self.records]
        self.edges = make_bins(self.values, 10, log_x=True)

    def test_matches_follow_bin_index(self):
        bins = [bin_index(self.edges, 30, True)]
        subset, gates = bins_subset(self.records, "context", self.edges,
                                    bins, log_x=True)
        self.assertEqual([r["in_tokens"] for r in subset], [30])
        self.assertEqual(gates, {"n_pool": 6, "n_matched": 1})

    def test_zero_clamps_into_first_bin(self):
        subset, _ = bins_subset(self.records, "context", self.edges, [0], True)
        vals = sorted(r["in_tokens"] for r in subset)
        self.assertIn(0, vals)   # the zero value clamps to 1 -> bin 0
        self.assertIn(1, vals)

    def test_multiple_disjoint_bins(self):
        b30 = bin_index(self.edges, 30, True)
        b500 = bin_index(self.edges, 500, True)
        subset, _ = bins_subset(self.records, "context", self.edges,
                                sorted({b30, b500}), True)
        self.assertEqual(sorted(r["in_tokens"] for r in subset), [30, 500])

    def test_empty_bins_and_bad_indices_raise(self):
        with self.assertRaises(ValueError):
            bins_subset(self.records, "context", self.edges, [], True)
        with self.assertRaises(ValueError):
            bins_subset(self.records, "context", self.edges, [99], True)

    def test_unknown_dim_raises(self):
        with self.assertRaises(KeyError):
            bins_subset(self.records, "contxt", self.edges, [0], True)


class TestBinHelpers(unittest.TestCase):
    def test_bin_runs(self):
        self.assertEqual(bin_runs([0, 1, 2, 5, 7, 8]), [(0, 2), (5, 5), (7, 8)])
        self.assertEqual(bin_runs([]), [])
        self.assertEqual(bin_runs([3, 3, 2]), [(2, 3)])  # dedup + sort

    def test_selection_to_bins(self):
        # a box from x=1.6 to x=4.2 covers bars 2, 3, 4 (bars sit at ints)
        self.assertEqual(selection_to_bins((1.6, 4.2), 60), [2, 3, 4])
        # a tiny box inside one bar's column selects exactly that bar
        self.assertEqual(selection_to_bins((2.3, 2.4), 60), [2])
        # clamped to the axis; fully off-axis selects nothing
        self.assertEqual(selection_to_bins((-5.0, 0.2), 60), [0])
        self.assertEqual(selection_to_bins((70.0, 80.0), 60), [])

    def test_bin_index_matches_counts_rules(self):
        edges = make_bins([1, 1000], 10, log_x=True)
        self.assertEqual(bin_index(edges, 0, True), 0)     # clamps up
        with self.assertRaises(ValueError):
            bin_index(edges, 10_000, True)                 # beyond last edge


if __name__ == "__main__":
    unittest.main()
