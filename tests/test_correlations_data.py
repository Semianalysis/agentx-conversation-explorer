"""Tests for the Correlations measure registry, exclusive-bin selection
store, and membership matching."""
import unittest

from modules.binning import bin_index, make_bins
from modules.theme import BRAND_PALETTE
from modules.correlations_data import (CHART_SLOTS, DEFAULT_AXES,
                                       MAX_SELECTIONS, MEASURES,
                                       assign_bin, bin_runs,
                                       clear_selection, initial_store,
                                       InertChart, PaletteExhausted,
                                       next_color_index, pick_bin,
                                       pick_range,
                                       measure_values, member_mask,
                                       selection_color)
from modules.figures import selection_to_bins


class TestMeasureRegistry(unittest.TestCase):
    def test_defaults_and_shapes(self):
        # per-chart X measures; y is always request count (user 2026-08-29)
        self.assertEqual(DEFAULT_AXES,
                         {"y1": "kv_cache_tokens", "y2": "new_input",
                          "y3": "decode_output"})
        for key in ("turn_number", "cumulative_time", "busy_time",
                    "kv_cache_tokens", "new_input", "decode_output"):
            self.assertIn(key, MEASURES)
        # hardware guessing removed 2026-08-30: no FLOPs / byte measures
        self.assertNotIn("turn_flops", MEASURES)
        self.assertNotIn("kv_cache_bytes", MEASURES)
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
        with self.assertRaises(KeyError):
            measure_values([{}], "kv_cache_bites")


class TestSelectionStore(unittest.TestCase):
    """A click IS the gesture: it takes the next brand color, or releases a
    bin it already owns."""

    def test_initial_shape_is_nothing_selected(self):
        s = initial_store()
        self.assertEqual(s["selections"], [])
        self.assertIsNone(s["live"])
        self.assertIsNone(s["chart"])

    def test_first_click_anchors_the_shared_chart(self):
        s = pick_bin(initial_store(), "y2", 5)
        self.assertEqual(s["chart"], "y2")
        self.assertEqual(s["selections"][0]["bins"], [5])

    def test_click_on_another_chart_raises_inert_not_the_cap(self):
        s = pick_bin(initial_store(), "y1", 5)
        with self.assertRaises(InertChart):
            pick_bin(s, "y3", 1)
        with self.assertRaises(InertChart):
            pick_range(s, "y2", 1, 3)
        # both stay ValueErrors, so the outer callback handler still sees them
        self.assertTrue(issubclass(InertChart, ValueError))
        self.assertTrue(issubclass(PaletteExhausted, ValueError))

    def test_each_click_takes_the_next_color(self):
        s = initial_store()
        for i, b in enumerate((1, 2, 3)):
            s = pick_bin(s, "y1", b)
            self.assertEqual(s["selections"][i]["color"], i)
        colors = [selection_color(x) for x in s["selections"]]
        self.assertEqual(len(set(colors)), 3)       # all different
        self.assertEqual(colors[0], BRAND_PALETTE[0])

    def test_colors_cycle_through_the_seven(self):
        s = initial_store()
        for b in range(MAX_SELECTIONS):
            s = pick_bin(s, "y1", b)
        self.assertEqual([x["color"] for x in s["selections"]],
                         list(range(MAX_SELECTIONS)))
        self.assertIsNone(next_color_index(s))      # all seven in use
        # the cap must be SAYABLE: a distinct type, so the click handler
        # cannot swallow it the way it silences an inert chart
        with self.assertRaises(PaletteExhausted) as ctx:
            pick_bin(s, "y1", 99)
        self.assertIn("clear one", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, InertChart)

    def test_releasing_a_bin_frees_its_color_for_reuse(self):
        s = initial_store()
        s = pick_bin(s, "y1", 1)                    # color 0
        s = pick_bin(s, "y1", 2)                    # color 1
        s = pick_bin(s, "y1", 1)                    # release the first
        self.assertEqual([x["color"] for x in s["selections"]], [1])
        # the cycle continues from where it was, then wraps to the free color
        nxt = next_color_index(s)
        self.assertEqual(nxt, 2)
        # five more picks use 2..6; the freed color 0 is then next in line
        s2 = s
        for i in range(5):
            s2 = pick_bin(s2, "y1", 10 + i)
        self.assertEqual([x["color"] for x in s2["selections"]],
                         [1, 2, 3, 4, 5, 6])
        self.assertEqual(next_color_index(s2), 0)   # wrapped to the freed one
        s2 = pick_bin(s2, "y1", 99)
        self.assertEqual(s2["selections"][-1]["color"], 0)

    def test_reclick_releases_and_drops_the_empty_selection(self):
        s = pick_bin(initial_store(), "y1", 5)
        s = pick_bin(s, "y1", 5)
        self.assertEqual(s["selections"], [])       # nothing left to show
        self.assertIsNone(s["chart"])               # unanchored again
        self.assertIsNone(s["live"])

    def test_unanchored_after_release_accepts_any_chart(self):
        s = pick_bin(initial_store(), "y1", 5)
        s = pick_bin(s, "y1", 5)
        s = pick_bin(s, "y3", 2)
        self.assertEqual(s["chart"], "y3")

    def test_a_box_select_is_one_gesture_in_one_color(self):
        s = pick_bin(initial_store(), "y1", 3)
        s = pick_range(s, "y1", 2, 4)
        # the range took bin 3 from the first selection, which then vanished
        self.assertEqual(len(s["selections"]), 1)
        self.assertEqual(s["selections"][0]["bins"], [2, 3, 4])
        self.assertEqual(len({x["color"] for x in s["selections"]}), 1)

    def test_inverted_range_is_normalized(self):
        s = pick_range(initial_store(), "y1", 7, 4)
        self.assertEqual(s["selections"][0]["bins"], [4, 5, 6, 7])

    def test_strip_click_still_transfers_between_selections(self):
        s = pick_bin(initial_store(), "y1", 5)      # S1 owns 5
        s = pick_bin(s, "y1", 9)                    # S2 owns 9
        s = assign_bin(s, s["selections"][1]["sid"], "y1", 5)   # S2 takes 5
        self.assertEqual([x["bins"] for x in s["selections"]], [[5, 9]])

    def test_clear_drops_the_selection_and_unanchors_when_last(self):
        s = pick_bin(initial_store(), "y1", 5)
        sid = s["selections"][0]["sid"]
        s = clear_selection(s, sid)
        self.assertEqual(s["selections"], [])
        self.assertIsNone(s["chart"])

    def test_clear_keeps_the_anchor_while_another_selection_holds_bins(self):
        s = pick_bin(initial_store(), "y1", 5)
        s = pick_bin(s, "y1", 7)
        s = clear_selection(s, s["selections"][0]["sid"])
        self.assertEqual(s["chart"], "y1")
        self.assertEqual(len(s["selections"]), 1)

    def test_mutations_are_copy_on_write(self):
        before = pick_bin(initial_store(), "y1", 1)
        pick_bin(before, "y1", 2)
        self.assertEqual(len(before["selections"]), 1)
        pick_bin(before, "y1", 1)
        self.assertEqual(before["selections"][0]["bins"], [1])

    def test_colors_are_the_brand_palette_in_its_order(self):
        self.assertEqual(len(BRAND_PALETTE), 7)
        self.assertEqual(BRAND_PALETTE[0], "#0B86D1")   # S2 Blue
        s = pick_bin(initial_store(), "y1", 1)
        self.assertEqual(selection_color(s["selections"][0]), "#0B86D1")

    def test_chart_slots(self):
        self.assertEqual(CHART_SLOTS, ("y1", "y2", "y3"))
        with self.assertRaises(KeyError):
            pick_bin(initial_store(), "y9", 1)


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
