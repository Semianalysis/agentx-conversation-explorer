"""The working set's edges — the three defects Bugbot found in bld 41.

These are the cases where the UI can hand a tab something the data layer
cannot satisfy: a tick decided from a re-sorted list, a dataset ticked before
its traces exist, and a selection that outlived its dataset.
"""
import unittest

from modules import sources as src
from modules.explorer_callbacks import _tick_hint, _ticks_after_load
from modules.explorer_data import live_selection
from modules.records import split_conv_id


class TestTickAfterLoad(unittest.TestCase):
    """sources.add re-SORTS by (kind, label), so the list order says nothing
    about which dataset was just loaded — only the loader knows."""

    def test_add_does_not_append_at_the_end(self):
        after = src.add([src.make_source("zz-web", src.WEBSITE, "zz")],
                        src.make_source("aa-local", src.LOCAL, "aa"))
        self.assertEqual([s["slug"] for s in after], ["zz-web", "aa-local"])
        # and a website source sorts BEFORE an existing local one
        after2 = src.add(after, src.make_source("mm-web", src.WEBSITE, "mm"))
        self.assertNotEqual(after2[-1]["slug"], "mm-web")

    def test_first_load_ticks_itself(self):
        self.assertEqual(_ticks_after_load([], "ds"), ["ds"])
        self.assertEqual(_ticks_after_load(None, "ds"), ["ds"])

    def test_a_later_load_never_changes_what_is_charted(self):
        self.assertEqual(_ticks_after_load(["a"], "b"), ["a"])
        self.assertEqual(_ticks_after_load(["a", "b"], "c"), ["a", "b"])

    def test_the_hint_tells_the_user_to_tick_only_when_needed(self):
        self.assertEqual(_tick_hint([]), "")
        self.assertIn("tick it", _tick_hint(["a"]))


class TestLiveSelection(unittest.TestCase):
    """A stale selection must never raise inside a chart callback, and must
    never be read as 'select nothing'."""

    POOL = ["dsA::c1", "dsA::c2", "dsB::c1"]

    def test_empty_selection_means_all(self):
        for sel in (None, {}, {"conv_ids": []}):
            self.assertEqual(live_selection(sel, self.POOL), [])

    def test_keeps_only_what_the_pool_holds(self):
        self.assertEqual(
            live_selection({"conv_ids": ["dsA::c2", "gone::c9"]}, self.POOL),
            ["dsA::c2"])

    def test_a_selection_matching_nothing_is_ignored_not_zeroed(self):
        # [] is read as "all conversations" by every consumer
        self.assertEqual(live_selection({"conv_ids": ["gone::c9"]}, self.POOL),
                         [])

    def test_accepts_any_id_container(self):
        self.assertEqual(
            live_selection({"conv_ids": ["dsB::c1"]}, {"dsB::c1": 1}),
            ["dsB::c1"])


class TestSelectionSurvivesWorkingSetChange(unittest.TestCase):
    """The rule pick_selection applies when the working set changes: ids are
    namespaced, so what survives is decidable from the slugs alone — no table
    needed, and the table it had was the previous one."""

    @staticmethod
    def _surviving(conv_ids, slugs):
        live = set(slugs)
        return {cid for cid in conv_ids if split_conv_id(cid)[0] in live}

    def test_unticking_a_dataset_drops_its_conversations(self):
        sel = {"dsA::c1", "dsA::c2", "dsB::c1"}
        self.assertEqual(self._surviving(sel, ["dsA"]), {"dsA::c1", "dsA::c2"})

    def test_unticking_everything_clears_the_selection(self):
        self.assertEqual(self._surviving({"dsA::c1"}, []), set())

    def test_adding_a_dataset_keeps_the_existing_selection(self):
        sel = {"dsA::c1"}
        self.assertEqual(self._surviving(sel, ["dsA", "dsB"]), sel)


if __name__ == "__main__":
    unittest.main()
