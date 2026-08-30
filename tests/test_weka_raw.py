"""Tests for the raw-weka flattener (port of the site's weka-structure.ts)."""
import unittest

from modules.weka_raw import build_conversation_structure, count_seen_prefix_blocks


def _req(t, in_t, out, hash_ids, api_time=1.0, model="m"):
    return {"t": t, "type": "n", "model": model, "in": in_t, "out": out,
            "hash_ids": hash_ids, "api_time": api_time}


class TestPrefixCacheSplit(unittest.TestCase):
    def test_contiguous_prefix_only(self):
        self.assertEqual(count_seen_prefix_blocks([1, 2, 9, 3], {1, 2, 3}), 2)
        self.assertEqual(count_seen_prefix_blocks([9, 1], {1}), 0)

    def test_split_and_partial_block_clamp(self):
        conv = {"id": "c", "block_size": 64, "requests": [
            _req(0.0, 130, 5, [1, 2, 3]),       # first: nothing cached
            _req(2.0, 190, 5, [1, 2, 3, 4]),    # 3 seen blocks=192 > in=190
        ]}
        s = build_conversation_structure(conv)
        n0, n1 = s["nodes"]
        self.assertEqual((n0["cached"], n0["uncached"]), (0, 130))
        self.assertEqual((n1["cached"], n1["uncached"]), (190, 0))  # clamped
        t = s["totals"]
        self.assertEqual(t["in"], t["cached"] + t["uncached"])

    def test_no_hash_ids_means_all_uncached(self):
        conv = {"id": "c", "requests": [_req(0.0, 100, 1, [])]}
        n = build_conversation_structure(conv)["nodes"][0]
        self.assertEqual((n["cached"], n["uncached"]), (0, 100))


class TestSubagentSemantics(unittest.TestCase):
    def test_child_seen_is_snapshot_not_merged_back(self):
        conv = {"id": "c", "block_size": 64, "requests": [
            _req(0.0, 128, 1, [1, 2]),
            {"type": "subagent", "t": 5.0, "agent_id": "a1",
             "requests": [_req(0.1, 192, 1, [1, 2, 9])]},  # sees parent's 1,2
            _req(9.0, 128, 1, [9, 8]),  # 9 must NOT be cached (no merge back)
        ]}
        s = build_conversation_structure(conv)
        sub = s["nodes"][1]
        self.assertEqual(sub["children"][0]["cached"], 128)  # 2 blocks seen
        last = s["nodes"][2]
        self.assertEqual(last["cached"], 0)

    def test_legacy_relative_child_time_and_group_range(self):
        conv = {"id": "c", "requests": [
            {"type": "subagent", "t": 100.0, "subagent_type": "Task",
             "requests": [_req(0.5, 64, 1, [1], api_time=2.0)]},
        ]}
        sub = build_conversation_structure(conv)["nodes"][0]
        child = sub["children"][0]
        self.assertEqual(child["startS"], 100.5)  # 0.5 < group t -> relative
        self.assertEqual(child["endS"], 102.5)
        self.assertEqual(sub["label"], "Task")
        self.assertEqual(sub["endS"], 102.5)  # no duration_ms -> child ends

    def test_counts_and_missing_requests_raise(self):
        conv = {"id": "c", "requests": [
            _req(0.0, 64, 1, [1]),
            {"type": "subagent", "t": 1.0, "requests": [_req(0.1, 64, 1, [2])]},
        ]}
        t = build_conversation_structure(conv)["totals"]
        self.assertEqual((t["numTurns"], t["numSubagentGroups"]), (1, 1))
        with self.assertRaises(ValueError):
            build_conversation_structure({"id": "bad"})


class TestGatingHelpers(unittest.TestCase):
    def test_internal_mode_env(self):
        import os
        from modules.hf_client import internal_mode
        for v, want in (("", False), ("0", False), ("false", False),
                        ("1", True), ("yes", True)):
            os.environ["AGENTX_INTERNAL"] = v
            self.assertEqual(internal_mode(), want, v)
        del os.environ["AGENTX_INTERNAL"]
        self.assertFalse(internal_mode())

    def test_slug_and_family(self):
        from modules.hf_client import hf_slug, is_weka_family
        self.assertEqual(hf_slug("semianalysisai/cc-traces-weka-051826"),
                         "hf--cc-traces-weka-051826")
        self.assertTrue(is_weka_family("semianalysisai/cc-traces-0"))
        self.assertFalse(is_weka_family("semianalysisai/multiturn-benchmark-data"))


if __name__ == "__main__":
    unittest.main()
