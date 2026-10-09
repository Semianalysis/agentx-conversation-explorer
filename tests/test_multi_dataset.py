"""Several datasets in one working set.

The working set is a LIST of slugs everywhere; conversation ids are namespaced
'<slug>::<conv_id>' so two datasets can be charted together without their
ordinals or ids colliding.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules import api_client, records
from modules.explorer_data import build_conversation_table
from modules.pause_callbacks import _pauses_for


def _turn(i=0, in_t=100, cached=60, uncached=40, out=10, start=0.0, end=1.0):
    return {"kind": "turn", "model": "m", "in": in_t, "cached": cached,
            "uncached": uncached, "out": out, "startS": start, "endS": end,
            "turnIndex": i}


class _Cache:
    """Two tiny cached datasets on disk, with api_client pointed at them."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.index = {}
        for slug, convs in (("dsA", ["c1", "c2"]), ("dsB", ["c1"])):
            d = root / slug / "conversations"
            d.mkdir(parents=True)
            for n, cid in enumerate(convs):
                (d / f"{cid}.json").write_text(json.dumps({
                    "conv_id": cid,
                    "structure": {"nodes": [
                        _turn(0, start=0.0, end=1.0),
                        # a 100 s pause, then the next turn
                        _turn(1, start=101.0, end=102.0)]},
                    # only dsA carries tool activity intervals
                    **({"activities": [{"category": "bash", "start_s": 1.0,
                                        "end_s": 41.0}]} if slug == "dsA" else {}),
                }), encoding="utf-8")
            self.index[slug] = [{"conv_id": c} for c in convs]
        self.patches = [
            mock.patch.object(api_client, "DATA_DIR", root),
            mock.patch.object(api_client, "fetch_conversation_index",
                              side_effect=lambda slug, **k: self.index[slug]),
        ]
        for p in self.patches:
            p.start()
        records.clear_pools()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        records.clear_pools()
        self.tmp.cleanup()
        return False


class TestNamespacing(unittest.TestCase):
    def test_split_round_trip(self):
        self.assertEqual(records.split_conv_id(records.namespaced("ds", "c1")),
                         ("ds", "c1"))

    def test_bare_id_raises(self):
        with self.assertRaises(ValueError):
            records.split_conv_id("c1")


class TestPool(unittest.TestCase):
    def test_pool_merges_and_namespaces(self):
        with _Cache():
            pool = records.load_pool(["dsA", "dsB"])
            self.assertEqual(len(pool), 6)       # 3 conversations x 2 turns
            self.assertEqual({r["dataset"] for r in pool}, {"dsA", "dsB"})
            # same conv_id 'c1' in both datasets stays distinct
            self.assertIn("dsA::c1", {r["conv_id"] for r in pool})
            self.assertIn("dsB::c1", {r["conv_id"] for r in pool})
            self.assertEqual(len({r["uid"] for r in pool}), 6)

    def test_empty_working_set_is_not_an_error(self):
        with _Cache():
            self.assertEqual(records.load_pool([]), [])
            self.assertEqual(records.merged_index([]), [])

    def test_single_dataset_pool_is_still_namespaced(self):
        with _Cache():
            pool = records.load_pool(["dsB"])
            self.assertEqual({r["conv_id"] for r in pool}, {"dsB::c1"})

    def test_merged_index_covers_the_pool_so_the_table_builds(self):
        with _Cache():
            pool = records.load_pool(["dsA", "dsB"])
            index = records.merged_index(["dsA", "dsB"])
            self.assertEqual(index, ["dsA::c1", "dsA::c2", "dsB::c1"])
            rows, _ = build_conversation_table(pool, index, None)
            # ordinals are per-dataset order, unique across the working set
            self.assertEqual([(r["id"], r["ordinal"]) for r in rows],
                             [("dsA::c1", 1), ("dsA::c2", 2), ("dsB::c1", 3)])

    def test_loading_one_dataset_of_a_pair_does_not_leak_the_other(self):
        with _Cache():
            self.assertEqual(len(records.load_pool(["dsB"])), 2)

    def test_activities_keyed_by_namespaced_id_only_where_present(self):
        with _Cache():
            acts = records.load_pool_activities(["dsA", "dsB"])
            self.assertEqual(set(acts), {"dsA::c1", "dsA::c2"})  # dsB has none


class TestPauseWorkingSet(unittest.TestCase):
    def test_empty_conv_id_list_means_every_conversation(self):
        """Regression: the Explorer writes conv_ids=[] whenever nothing is
        ticked. Reading that as 'select nothing' emptied the whole tab."""
        with _Cache():
            every = _pauses_for(["dsA"], None, 0.0)
            self.assertEqual(len(every), 2)           # one pause per conversation
            self.assertEqual(
                len(_pauses_for(["dsA"], {"slugs": ["dsA"], "conv_ids": []}, 0.0)),
                len(every))

    def test_a_real_selection_still_scopes(self):
        with _Cache():
            ps = _pauses_for(["dsA"], {"slugs": ["dsA"],
                                       "conv_ids": ["dsA::c1"]}, 0.0)
            self.assertEqual(len(ps), 1)
            self.assertEqual(ps[0]["cid"], "dsA::c1")

    def test_activities_reach_the_pauses_through_the_namespace(self):
        with _Cache():
            p = _pauses_for(["dsA"], None, 0.0)[0]
            self.assertAlmostEqual(p["duration_s"], 100.0)
            self.assertAlmostEqual(p["causes"]["bash"], 40.0)
            self.assertAlmostEqual(p["causes"]["unknown"], 60.0)
            self.assertAlmostEqual(sum(p["causes"].values()), p["duration_s"])

    def test_min_duration_filter(self):
        with _Cache():
            self.assertEqual(_pauses_for(["dsA"], None, 1000.0), [])

    def test_working_set_of_two_pools_both_datasets_pauses(self):
        with _Cache():
            self.assertEqual(len(_pauses_for(["dsA", "dsB"], None, 0.0)), 3)


class TestLaneRule(unittest.TestCase):
    """The list, the growth chart and the pause timeline must agree about
    what a conversation contains - they used to disagree for OpenClaw
    subagent lanes (573 listed, 479 drawn)."""

    def _sub(self, cid, i, start, end):
        r = _turn(i, start=start, end=end)
        return {"uid": f"{cid}#{i}", "conv_id": cid, "role": "subagent",
                "agent_id": "sa", "depth": 1, "model": "m",
                "in_tokens": r["in"], "cached_tokens": r["cached"],
                "uncached_tokens": r["uncached"], "out_tokens": r["out"],
                "start_s": start, "end_s": end, "turn_index": i}

    def test_a_subagent_only_lane_is_its_own_lane(self):
        recs = [self._sub("c", 1, 10, 11), self._sub("c", 0, 0, 1)]
        self.assertFalse(records.has_main_agent(recs))
        self.assertEqual([r["turn_index"] for r in records.lane_turns(recs)],
                         [0, 1])

    def test_main_turns_win_when_there_are_any(self):
        recs = [self._sub("c", 0, 0, 1),
                {**self._sub("c", 1, 2, 3), "role": "main", "depth": 0}]
        self.assertEqual([r["role"] for r in records.lane_turns(recs)], ["main"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            records.lane_turns([])

    def test_every_listed_conversation_has_a_curve(self):
        from modules.explorer_data import conversation_curves
        recs = [self._sub("sub", 0, 0, 1), self._sub("sub", 1, 10, 11),
                {**self._sub("main", 0, 0, 1), "role": "main", "depth": 0}]
        rows, sub_only = build_conversation_table(recs, ["sub", "main"], None)
        self.assertEqual(sub_only, 1)
        curves = conversation_curves(recs)
        self.assertEqual(set(curves), {r["id"] for r in rows})


if __name__ == "__main__":
    unittest.main()
