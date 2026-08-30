"""Tests for the Correlations measure registry, exclusive-bin selection
store, and membership matching."""
import unittest

from modules.binning import bin_index, make_bins
from modules.correlations_data import (CHART_SLOTS, DEFAULT_AXES,
                                       MAX_SELECTIONS, MEASURES, add_selection,
                                       assign_bin, assign_bin_range, bin_runs,
                                       clear_selection, initial_store,
                                       measure_values, member_mask,
                                       selection_color, set_live)
from modules.figures import selection_to_bins


class TestMeasureRegistry(unittest.TestCase):
    def test_defaults_and_shapes(self):
        # per-chart X measures; y is always request count (user 2026-08-29)
        self.assertEqual(DEFAULT_AXES,
                         {"y1": "kv_cache_tokens", "y2": "new_input",
                          "y3": "decode_output"})
        for key in ("turn_number", "cumulative_time", "busy_time",
                    "kv_cache_tokens", "kv_cache_bytes", "new_input",
                    "decode_output", "turn_flops"):
            self.assertIn(key, MEASURES)
        for key, m in MEASURES.items():
            self.assertTrue(m.get("info", "").strip(), f"{key} missing info")
        self.assertEqual(MEASURES["new_input"]["label"],
                         "uncached input (tokens)")

    def test_getters_read_enriched_rows(self):
        row = {"seq": 7, "start_s": 100.0, "busy_s": 40.0, "kv_bytes": 5e9,
               "in_tokens": 2000, "cached_tokens": 1877,
               "uncached_tokens": 123, "out_tokens": 45, "flops": 1e12}
        self.assertEqual(measure_values([row], "kv_cache_tokens"), [1877])
        self.assertEqual(measure_values([row], "turn_number"), [7])
        self.assertEqual(measure_values([row], "busy_time"), [40.0])
        self.assertEqual(measure_values([row], "turn_flops"), [1e12])
        with self.assertRaises(KeyError):
            measure_values([{}], "kv_cache_bites")


class TestSelectionStore(unittest.TestCase):
    def test_initial_shape(self):
        s = initial_store()
        self.assertEqual(len(s["selections"]), 1)  # one section always present
        self.assertEqual(s["live"], s["selections"][0]["sid"])
        self.assertIsNone(s["chart"])

    def test_first_pick_anchors_the_shared_chart(self):
        s = assign_bin(initial_store(), 0, "y2", 5)
        self.assertEqual(s["chart"], "y2")
        self.assertEqual(s["selections"][0]["bins"], [5])

    def test_click_on_other_chart_raises(self):
        s = assign_bin(initial_store(), 0, "y1", 5)
        with self.assertRaises(ValueError):
            assign_bin(s, 0, "y3", 1)

    def test_bins_are_exclusive_between_inspectors(self):
        s = assign_bin(initial_store(), 0, "y1", 5)
        s = add_selection(s)                    # S2 armed
        s = assign_bin(s, 1, "y1", 5)           # steals bin 5 from S1
        self.assertEqual(s["selections"][0]["bins"], [])
        self.assertEqual(s["selections"][1]["bins"], [5])

    def test_reclick_own_bin_releases_it(self):
        s = assign_bin(initial_store(), 0, "y1", 5)
        s = assign_bin(s, 0, "y1", 5)
        self.assertEqual(s["selections"][0]["bins"], [])
        self.assertIsNone(s["chart"])           # all empty -> unanchored

    def test_unanchored_after_release_accepts_any_chart(self):
        s = assign_bin(initial_store(), 0, "y1", 5)
        s = assign_bin(s, 0, "y1", 5)           # release -> unanchored
        s = assign_bin(s, 0, "y3", 2)           # new anchor allowed
        self.assertEqual(s["chart"], "y3")

    def test_range_claims_and_steals_but_never_releases(self):
        s = assign_bin(initial_store(), 0, "y1", 3)
        s = add_selection(s)
        s = assign_bin_range(s, 1, "y1", 2, 4)
        self.assertEqual(s["selections"][0]["bins"], [])
        self.assertEqual(s["selections"][1]["bins"], [2, 3, 4])
        s = assign_bin_range(s, 1, "y1", 3, 3)  # re-select own range: no-op
        self.assertEqual(s["selections"][1]["bins"], [2, 3, 4])

    def test_add_selection_arms_it_and_caps(self):
        s = add_selection(initial_store())
        self.assertEqual(s["live"], s["selections"][-1]["sid"])
        while len(s["selections"]) < MAX_SELECTIONS:
            s = add_selection(s)
        with self.assertRaises(ValueError):
            add_selection(s)

    def test_clear_keeps_section_and_unanchors_when_last(self):
        s = assign_bin(initial_store(), 0, "y1", 5)
        s = clear_selection(s, 0)
        self.assertEqual(len(s["selections"]), 1)   # never disappears
        self.assertEqual(s["selections"][0]["bins"], [])
        self.assertIsNone(s["chart"])
        # anchor survives while ANOTHER selection still holds bins
        s = assign_bin(s, 0, "y1", 5)
        s = add_selection(s)
        s = assign_bin(s, 1, "y1", 7)
        s = clear_selection(s, 0)
        self.assertEqual(s["chart"], "y1")

    def test_set_live_arms(self):
        s = add_selection(initial_store())
        s = set_live(s, 0)
        self.assertEqual(s["live"], 0)
        with self.assertRaises(KeyError):
            set_live(s, 42)

    def test_mutations_are_copy_on_write(self):
        before = initial_store()
        assign_bin(before, 0, "y1", 1)
        self.assertEqual(before["selections"][0]["bins"], [])
        self.assertIsNone(before["chart"])

    def test_selection_colors_stable_and_distinct(self):
        self.assertEqual(selection_color(3), selection_color(3))
        self.assertNotEqual(selection_color(0), selection_color(1))

    def test_chart_slots(self):
        self.assertEqual(CHART_SLOTS, ("y1", "y2", "y3"))
        with self.assertRaises(KeyError):
            assign_bin(initial_store(), 0, "y9", 1)


class TestMemberMask(unittest.TestCase):
    def setUp(self):
        self.values = [1.0, 2.0, 30.0, 500.0, 999.0, 0.0]
        self.edges = make_bins(self.values, 10, log_x=True)

    def test_mask_follows_bin_index(self):
        b30 = bin_index(self.edges, 30.0, True)
        mask = member_mask(self.values, self.edges, [b30], True)
        self.assertEqual([v for v, m in zip(self.values, mask) if m], [30.0])

    def test_zero_clamps_into_first_bin(self):
        mask = member_mask(self.values, self.edges, [0], True)
        matched = sorted(v for v, m in zip(self.values, mask) if m)
        self.assertIn(0.0, matched)
        self.assertIn(1.0, matched)

    def test_empty_bins_and_bad_indices_raise(self):
        with self.assertRaises(ValueError):
            member_mask(self.values, self.edges, [], True)
        with self.assertRaises(ValueError):
            member_mask(self.values, self.edges, [99], True)


class TestBinHelpers(unittest.TestCase):
    def test_bin_runs(self):
        self.assertEqual(bin_runs([0, 1, 2, 5, 7, 8]), [(0, 2), (5, 5), (7, 8)])
        self.assertEqual(bin_runs([]), [])
        self.assertEqual(bin_runs([3, 3, 2]), [(2, 3)])  # dedup + sort

    def test_selection_to_bins(self):
        self.assertEqual(selection_to_bins((1.6, 4.2), 60), [2, 3, 4])
        self.assertEqual(selection_to_bins((2.3, 2.4), 60), [2])
        self.assertEqual(selection_to_bins((-5.0, 0.2), 60), [0])
        self.assertEqual(selection_to_bins((70.0, 80.0), 60), [])


if __name__ == "__main__":
    unittest.main()
