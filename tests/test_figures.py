"""Contract tests for the Correlations figure builder (trace shapes)."""
import unittest

from modules.binning import make_bins
from modules.figures import multi_histogram_figure


def _fig(own_bins=None, overlays=None):
    values = list(range(1, 100))
    edges = make_bins(values, 10, log_x=False)
    heights = [10.0] * 10
    return multi_histogram_figure(
        edges, heights, "t", "x", "requests", False, False,
        own_marks=[(own_bins, "#123456", "S1")] if own_bins else [],
        overlays=overlays or [])


class TestMultiHistogramFigure(unittest.TestCase):
    def test_trace_order_and_click_target(self):
        fig = _fig(own_bins=[2], overlays=[([1.0] * 10, "#123456", "S1")])
        kinds = [type(t).__name__ for t in fig.data]
        self.assertEqual(kinds[0], "Scatter")          # silhouette first
        self.assertEqual(kinds[-1], "Bar")             # click target last
        self.assertEqual(fig.data[-1].marker.color, "rgba(0,0,0,0)")
        self.assertEqual(fig.layout.barmode, "overlay")

    def test_own_marks_span_all_bins_with_zero_gaps(self):
        # regression: sparse x positions made plotly auto-derive bar WIDTH
        # from the gap between selected bins (giant slabs between bins 2 & 8)
        fig = _fig(own_bins=[2, 8])
        own = next(t for t in fig.data if t.name == "S1 bins")
        self.assertEqual(list(own.x), list(range(10)))  # full length
        self.assertEqual([y for y in own.y],
                         [0, 0, 10.0, 0, 0, 0, 0, 0, 10.0, 0])

    def test_overlays_stack_via_explicit_bases(self):
        fig = _fig(overlays=[([1.0] * 10, "#111111", "S1"),
                             ([2.0] * 10, "#222222", "S2")])
        s1 = next(t for t in fig.data if t.name == "S1")
        s2 = next(t for t in fig.data if t.name == "S2")
        self.assertEqual(list(s1.base), [0.0] * 10)
        self.assertEqual(list(s2.base), [1.0] * 10)     # sits on S1

    def test_mismatched_heights_raise(self):
        values = list(range(1, 100))
        edges = make_bins(values, 10, log_x=False)
        with self.assertRaises(ValueError):
            multi_histogram_figure(edges, [1.0] * 9, "t", "x", "y", False,
                                   False, own_marks=[], overlays=[])
        with self.assertRaises(ValueError):
            multi_histogram_figure(edges, [1.0] * 10, "t", "x", "y", False,
                                   False, own_marks=[],
                                   overlays=[([1.0] * 9, "#111111", "S1")])


if __name__ == "__main__":
    unittest.main()
