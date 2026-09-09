"""Idle-gap measure (undefined for a conversation's first turn) and the
one-measure histogram binning rule."""
import unittest

from modules.arch import ARCHITECTURES, resolve_assumptions
from modules.binning import bin_counts, histogram_bins, make_bins
from modules.correlations_data import MEASURES as CORR_MEASURES
from modules.correlations_data import measure_values as corr_values
from modules.correlations_data import member_mask
from modules.deepdive_data import aggregate_selection, grouped_series
from modules.figures import simple_histogram_figure
from modules.measures import Y_MEASURES, measure_series


def _rec(conv_id, i, start, end, role="main"):
    return {"uid": f"{conv_id}#{role}{i}", "conv_id": conv_id, "role": role,
            "agent_id": None, "depth": 0, "model": "m",
            "in_tokens": 1000, "cached_tokens": 900, "uncached_tokens": 100,
            "out_tokens": 50, "start_s": start, "end_s": end, "turn_index": i}


def _agg(pool, ids, ordinals):
    arch_key, _, cfg = resolve_assumptions(None)
    return aggregate_selection(pool, ids, ARCHITECTURES[arch_key], cfg,
                               ordinals)


class TestIdleGapEnrichment(unittest.TestCase):
    def test_first_turn_undefined_then_gap_to_previous_end(self):
        pool = [_rec("cA", 0, 0.0, 10.0),      # first: no previous turn
                _rec("cA", 1, 25.0, 30.0),     # 15 s idle after 10.0
                _rec("cA", 2, 30.0, 31.0)]     # back-to-back: 0 s
        rows = _agg(pool, ["cA"], {"cA": 1})["per_request"]
        self.assertEqual([r["idle_gap_s"] for r in rows], [None, 15.0, 0.0])

    def test_overlapping_subagent_reads_zero_and_frontier_is_the_max_end(self):
        pool = [_rec("cA", 0, 0.0, 100.0),                     # long main turn
                _rec("cA", 1, 20.0, 30.0, role="subagent"),    # inside it
                _rec("cA", 2, 130.0, 131.0)]                   # 30 s after 100
        rows = _agg(pool, ["cA"], {"cA": 1})["per_request"]
        gaps = [r["idle_gap_s"] for r in rows]
        self.assertIsNone(gaps[0])
        self.assertEqual(gaps[1], 0.0)    # started while the main turn ran
        self.assertEqual(gaps[2], 30.0)   # from the frontier (100), not 30

    def test_resets_per_conversation(self):
        pool = [_rec("cA", 0, 0.0, 5.0), _rec("cA", 1, 10.0, 11.0),
                _rec("cB", 0, 0.0, 5.0)]
        rows = _agg(pool, ["cA", "cB"], {"cA": 1, "cB": 2})["per_request"]
        first_b = [r for r in rows if r["cid"] == "cB"][0]
        self.assertIsNone(first_b["idle_gap_s"])


class TestIdleGapMeasures(unittest.TestCase):
    def setUp(self):
        pool = [_rec("cA", 0, 0.0, 10.0), _rec("cA", 1, 25.0, 30.0),
                _rec("cB", 0, 0.0, 1.0), _rec("cB", 1, 1.5, 2.0)]
        self.rows = _agg(pool, ["cA", "cB"], {"cA": 1, "cB": 2})["per_request"]

    def test_registered_on_both_tabs_with_help(self):
        for reg in (Y_MEASURES, CORR_MEASURES):
            self.assertIn("idle_gap", reg)
            self.assertTrue(reg["idle_gap"]["info"].strip())

    def test_undefined_first_turns_stay_none_not_zero(self):
        deep = measure_series(self.rows, "idle_gap")
        self.assertEqual(deep.count(None), 2)          # one per conversation
        self.assertNotIn(0.0, deep)                    # never faked as zero
        self.assertIn(15.0, deep)                      # cA's real gap
        self.assertEqual(min(v for v in deep if v is not None), 1.0)  # 0.5->1
        corr = corr_values(self.rows, "idle_gap")
        self.assertEqual(corr.count(None), 2)
        self.assertEqual(corr[3], 0.5)                 # corr keeps raw seconds

    def test_grouped_series_drops_undefined_rows(self):
        g = grouped_series(self.rows, "turn_number", "idle_gap")
        self.assertEqual(g["xs"], [2])                 # turn 1 has no gaps
        self.assertEqual(g["n_alive"], [2])

    def test_member_mask_never_matches_undefined(self):
        vals = corr_values(self.rows, "idle_gap")
        edges = make_bins([v for v in vals if v is not None], 10, log_x=True)
        mask = member_mask(vals, edges, [0], log_x=True)
        self.assertFalse(mask[0])                      # None is in no bin
        self.assertFalse(mask[2])


class TestHistogramBins(unittest.TestCase):
    def test_continuous_measure_gets_n_bins(self):
        values = [1.0, 2.5, 900.25, 10_000.5]
        self.assertEqual(len(histogram_bins(values, log_x=True)) - 1, 100)
        self.assertEqual(len(histogram_bins(values, log_x=False)) - 1, 100)

    def test_small_range_ordinal_gets_one_bar_per_value(self):
        edges = histogram_bins([1, 2, 3, 3, 7], log_x=False)
        self.assertEqual(len(edges) - 1, 7)            # values 1..7
        counts = bin_counts([1, 2, 3, 3, 7], edges, log_x=False)
        self.assertEqual(counts, [1, 1, 2, 0, 0, 0, 1])
        self.assertEqual(sum(counts), 5)               # max lands in last bin

    def test_wide_integer_range_falls_back_to_n_bins(self):
        values = list(range(1, 5000))
        self.assertEqual(len(histogram_bins(values, log_x=True)) - 1, 100)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            histogram_bins([], log_x=True)


class TestSimpleHistogramFigure(unittest.TestCase):
    def test_bars_and_count_mismatch(self):
        edges = histogram_bins([1, 2, 3], log_x=False)
        counts = bin_counts([1, 2, 3], edges, log_x=False)
        fig = simple_histogram_figure(edges, counts, "t", "m", False)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(list(fig.data[0].y), counts)
        self.assertEqual(fig.layout.yaxis.title.text, "requests")
        with self.assertRaises(ValueError):
            simple_histogram_figure(edges, counts[:-1], "t", "m", False)


if __name__ == "__main__":
    unittest.main()
