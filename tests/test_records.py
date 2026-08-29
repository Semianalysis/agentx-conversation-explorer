"""Contract + invariant tests for record flattening."""
import unittest

from modules.records import flatten_conversation, pool_models


def _turn(i=0, model="claude-opus-4-8", in_t=100, cached=60, uncached=40, out=10,
          start=0.0, end=1.0):
    return {"kind": "turn", "model": model, "in": in_t, "cached": cached,
            "uncached": uncached, "out": out, "startS": start, "endS": end,
            "turnIndex": i}


class TestFlatten(unittest.TestCase):
    def test_main_turns_flatten(self):
        recs = flatten_conversation("c1", {"nodes": [_turn(0), _turn(1)]})
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["uid"], "c1#0")
        self.assertEqual(recs[0]["role"], "main")
        self.assertEqual(recs[0]["depth"], 0)
        self.assertEqual(recs[0]["in_tokens"], 100)
        self.assertEqual(recs[0]["cached_tokens"], 60)
        self.assertEqual(recs[0]["uncached_tokens"], 40)
        self.assertEqual(recs[0]["out_tokens"], 10)

    def test_subagent_children_emitted_aggregate_not(self):
        structure = {"nodes": [
            _turn(0),
            {"kind": "subagent", "agentId": "sa_1",
             "in": 999999, "out": 999999,  # aggregate numbers must NOT be emitted
             "children": [_turn(0, model="claude-haiku-4-5-20251001")]},
        ]}
        recs = flatten_conversation("c1", structure)
        self.assertEqual(len(recs), 2)
        sub = recs[1]
        self.assertEqual(sub["role"], "subagent")
        self.assertEqual(sub["agent_id"], "sa_1")
        self.assertEqual(sub["depth"], 1)
        self.assertEqual(sub["uid"], "c1#1.0")
        self.assertEqual(sub["in_tokens"], 100)  # child's tokens, not aggregate

    def test_nested_subagent(self):
        structure = {"nodes": [
            {"kind": "subagent", "agentId": "outer", "children": [
                _turn(0),
                {"kind": "subagent", "agentId": "inner", "children": [_turn(0)]},
            ]},
        ]}
        recs = flatten_conversation("c1", structure)
        self.assertEqual([r["depth"] for r in recs], [1, 2])
        self.assertEqual(recs[1]["agent_id"], "inner")

    def test_uid_uniqueness(self):
        structure = {"nodes": [
            _turn(0),
            {"kind": "subagent", "agentId": "a", "children": [_turn(0), _turn(1)]},
            _turn(2),
        ]}
        recs = flatten_conversation("c1", structure)
        uids = [r["uid"] for r in recs]
        self.assertEqual(len(uids), len(set(uids)))

    def test_cached_plus_uncached_invariant_raises(self):
        bad = _turn()
        bad["cached"] = 999  # 999 + 40 != 100
        with self.assertRaises(ValueError) as ctx:
            flatten_conversation("c1", {"nodes": [bad]})
        self.assertIn("in == cached + uncached", str(ctx.exception))

    def test_missing_field_raises(self):
        bad = _turn()
        del bad["model"]
        with self.assertRaises(ValueError):
            flatten_conversation("c1", {"nodes": [bad]})

    def test_negative_tokens_raise(self):
        bad = _turn(in_t=-5, cached=-10, uncached=5)
        with self.assertRaises(ValueError):
            flatten_conversation("c1", {"nodes": [bad]})

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError) as ctx:
            flatten_conversation("c1", {"nodes": [{"kind": "mystery"}]})
        self.assertIn("mystery", str(ctx.exception))

    def test_empty_structure_raises(self):
        with self.assertRaises(ValueError):
            flatten_conversation("c1", {"nodes": []})

    def test_token_fields_exist_on_records(self):
        recs = flatten_conversation("c1", {"nodes": [_turn(0)]})
        for field in ("in_tokens", "uncached_tokens", "out_tokens"):
            self.assertIn(field, recs[0])

    def test_pool_models_sorted_by_count(self):
        recs = flatten_conversation("c1", {"nodes": [
            _turn(0, model="b"), _turn(1, model="a"), _turn(2, model="a")]})
        self.assertEqual(pool_models(recs), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
