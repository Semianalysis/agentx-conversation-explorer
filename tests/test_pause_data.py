"""Pause model: exact decomposition, lane rules, binning, ISL stats."""
import unittest

from modules.pause_data import (SUBAGENT, UNKNOWN, all_pauses, cause_totals,
                                conversation_pauses, decompose, duration_bins,
                                isl_by_bin, stack_by_cause)


def _rec(conv_id, uid, start, end, role="main", in_t=1000, unc=100, out=50):
    return {"uid": uid, "conv_id": conv_id, "role": role, "agent_id": None,
            "depth": 0 if role == "main" else 1, "model": "m",
            "in_tokens": in_t, "cached_tokens": in_t - unc,
            "uncached_tokens": unc, "out_tokens": out,
            "start_s": start, "end_s": end, "turn_index": 0}


class TestDecompose(unittest.TestCase):
    def test_partition_is_exact(self):
        d = decompose(0, 100, [("tool", 10, 40), (SUBAGENT, 60, 70)])
        self.assertEqual(d["tool"], 30)
        self.assertEqual(d[SUBAGENT], 10)
        self.assertEqual(d[UNKNOWN], 60)
        self.assertAlmostEqual(sum(d.values()), 100)

    def test_overlaps_are_claimed_once_first_label_wins(self):
        d = decompose(0, 10, [("a", 0, 6), ("b", 4, 10)])
        self.assertEqual(d["a"], 6)
        self.assertEqual(d["b"], 4)          # not 6 - the overlap is a's
        self.assertEqual(d[UNKNOWN], 0)
        self.assertAlmostEqual(sum(d.values()), 10)

    def test_intervals_are_clipped_to_the_pause(self):
        d = decompose(10, 20, [("a", 0, 15), ("b", 18, 100)])
        self.assertEqual((d["a"], d["b"], d[UNKNOWN]), (5, 2, 3))

    def test_fully_covered_pause_has_no_unknown(self):
        d = decompose(0, 5, [("a", -10, 10)])
        self.assertEqual(d[UNKNOWN], 0)

    def test_inverted_span_raises(self):
        with self.assertRaises(ValueError):
            decompose(10, 5, [])


class TestConversationPauses(unittest.TestCase):
    def test_gap_on_the_main_lane_with_subagent_cause(self):
        recs = [_rec("c", "m0", 0, 10),
                _rec("c", "s0", 20, 50, role="subagent"),
                _rec("c", "m1", 100, 110, in_t=9000, unc=800)]
        ps = conversation_pauses(recs)
        self.assertEqual(len(ps), 1)
        p = ps[0]
        self.assertEqual((p["start_s"], p["end_s"]), (10, 100))
        self.assertEqual(p["duration_s"], 90)
        self.assertEqual(p["causes"][SUBAGENT], 30)      # the subagent ran
        self.assertEqual(p["causes"][UNKNOWN], 60)       # the rest is honest
        self.assertEqual(p["next_in_tokens"], 9000)      # ISL after the pause
        self.assertEqual(p["next_uncached_tokens"], 800)

    def test_activities_name_the_cause(self):
        recs = [_rec("c", "m0", 0, 10), _rec("c", "m1", 100, 110)]
        acts = [{"category": "shell_code", "start_s": 12, "end_s": 42}]
        p = conversation_pauses(recs, acts)[0]
        self.assertEqual(p["causes"]["shell_code"], 30)
        self.assertEqual(p["causes"][UNKNOWN], 60)

    def test_back_to_back_and_overlapping_requests_make_no_pause(self):
        recs = [_rec("c", "a", 0, 10), _rec("c", "b", 10, 20),
                _rec("c", "c", 15, 30)]
        self.assertEqual(conversation_pauses(recs), [])

    def test_frontier_not_previous_end(self):
        # a long request followed by a short one: the gap is measured from
        # the furthest end reached, never from the last row's end
        recs = [_rec("c", "a", 0, 100), _rec("c", "b", 10, 20),
                _rec("c", "d", 150, 160)]
        ps = conversation_pauses(recs)
        self.assertEqual([p["duration_s"] for p in ps], [50])

    def test_subagent_only_conversation_uses_its_own_lane(self):
        recs = [_rec("c", "s0", 0, 10, role="subagent"),
                _rec("c", "s1", 40, 50, role="subagent")]
        ps = conversation_pauses(recs)
        self.assertEqual(len(ps), 1)
        self.assertEqual(ps[0]["duration_s"], 30)

    def test_single_request_has_no_pause_and_empty_raises(self):
        self.assertEqual(conversation_pauses([_rec("c", "a", 0, 1)]), [])
        with self.assertRaises(ValueError):
            conversation_pauses([])

    def test_all_pauses_scopes_to_selected_conversations(self):
        pool = [_rec("c1", "a", 0, 1), _rec("c1", "b", 10, 11),
                _rec("c2", "a", 0, 1), _rec("c2", "b", 20, 21)]
        self.assertEqual(len(all_pauses(pool)), 2)
        only = all_pauses(pool, {"c2"})
        self.assertEqual([p["cid"] for p in only], ["c2"])
        self.assertEqual(only[0]["duration_s"], 19)


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.pauses = [
            {"duration_s": 2, "next_in_tokens": 100,
             "causes": {"shell_code": 1.0, UNKNOWN: 1.0}},
            {"duration_s": 20, "next_in_tokens": 500,
             "causes": {"shell_code": 15.0, UNKNOWN: 5.0}},
            {"duration_s": 300, "next_in_tokens": 900,
             "causes": {UNKNOWN: 300.0}},
        ]

    def test_cause_totals_sorted_with_unknown_last(self):
        t = cause_totals(self.pauses)
        self.assertEqual(list(t), ["shell_code", UNKNOWN])
        self.assertEqual(t["shell_code"], 16.0)
        self.assertEqual(t[UNKNOWN], 306.0)

    def test_stack_time_vs_count(self):
        edges = duration_bins(self.pauses, 4, log_x=True)
        t = stack_by_cause(self.pauses, edges, True, ["shell_code", UNKNOWN],
                           "time")
        self.assertAlmostEqual(sum(t["shell_code"]) + sum(t[UNKNOWN]),
                               sum(p["duration_s"] for p in self.pauses))
        c = stack_by_cause(self.pauses, edges, True, ["shell_code", UNKNOWN],
                           "count")
        self.assertEqual(sum(c["shell_code"]) + sum(c[UNKNOWN]), 3)
        with self.assertRaises(ValueError):
            stack_by_cause(self.pauses, edges, True, [UNKNOWN], "sideways")

    def test_isl_bins_report_none_for_empty_bins(self):
        edges = duration_bins(self.pauses, 10, log_x=True)
        isl = isl_by_bin(self.pauses, edges, True, "next_in_tokens")
        self.assertEqual(len(isl["median"]), len(edges) - 1)
        self.assertIn(None, isl["median"])              # empty bins stay empty
        self.assertEqual([m for m in isl["median"] if m is not None],
                         sorted(m for m in isl["median"] if m is not None))
        self.assertEqual(sum(isl["n"]), 3)

    def test_bins_need_pauses(self):
        with self.assertRaises(ValueError):
            duration_bins([], 10, True)


if __name__ == "__main__":
    unittest.main()
