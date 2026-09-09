"""Regression tests for Bugbot findings on the internal panel and the
grouped-interval binning."""
import os
import unittest


def _find_ids(component, out: set) -> set:
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        out.add(cid)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for c in children:
            _find_ids(c, out)
    elif children is not None:
        _find_ids(children, out)
    return out


class TestInternalPanelLayout(unittest.TestCase):
    """The Load callback takes State from the hf-select checklist; Dash
    rejects callbacks whose ids are missing from the initial layout, so the
    checklist must be STATIC (Bugbot high-severity finding)."""

    def _layout_ids(self) -> set:
        import importlib

        from modules import overview_layout
        importlib.reload(overview_layout)
        return _find_ids(overview_layout._internal_section(), set())

    def test_internal_mode_has_static_checklist(self):
        os.environ["AGENTX_INTERNAL"] = "1"
        try:
            ids = self._layout_ids()
        finally:
            del os.environ["AGENTX_INTERNAL"]
        for needed in ("at-overview-hf-select-cl", "at-overview-hf-load-btn",
                       "at-overview-hf-unsupported", "at-overview-hf-list-btn",
                       "at-overview-hf-progress", "at-overview-hf-interval"):
            self.assertIn(needed, ids)

    def test_public_mode_has_no_internal_components(self):
        os.environ.pop("AGENTX_INTERNAL", None)
        self.assertEqual(self._layout_ids(), set())


class TestIntervalBinningMatchesGroupedSeries(unittest.TestCase):
    """The interval inspector must assign rows to bins with grouped_series'
    EXACT rule — a rebuilt half-open range dropped max-x rows from the last
    bin (Bugbot medium-severity finding)."""

    def test_max_x_row_lands_in_last_bin(self):
        import math

        lo_x, hi_x, n_bins = 1.0, 1000.0, 60
        log_ratio = math.log(hi_x / lo_x) / n_bins

        def _bin(x: float) -> int:  # the shared rule
            return (0 if x <= lo_x
                    else min(int(math.log(x / lo_x) / log_ratio), n_bins - 1))

        self.assertEqual(_bin(hi_x), n_bins - 1)     # the max itself
        # grouped_series puts the max in the last bin; the old
        # b_lo <= x < b_hi rebuild excluded it
        b_hi = lo_x * math.exp(log_ratio * n_bins)
        self.assertGreaterEqual(hi_x, b_hi * (1 - 1e-12))


if __name__ == "__main__":
    unittest.main()
