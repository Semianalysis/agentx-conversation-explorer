"""Contract tests for the OpenClaw trajectory importer."""
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from modules.openclaw_import import (build_conversations, import_openclaw,
                                     read_steps)

COLS = ["session_id", "run_id", "session_key", "kind", "provider", "model",
        "api", "ts_end", "step_latency_s", "prev_role", "input", "output",
        "cacheRead", "cacheWrite", "context_tokens", "reasoning_tokens",
        "n_tool_calls", "tool_names", "text_chars", "stop"]


def _row(sid, ts, lat, inp, out, cr, cw, kind="chat_web", model="m1",
         key="agent:main:main"):
    d = dict.fromkeys(COLS, "")
    d.update(session_id=sid, session_key=key, kind=kind, model=model,
             ts_end=ts, step_latency_s=lat, input=inp, output=out,
             cacheRead=cr, cacheWrite=cw,
             context_tokens=sum(v for v in (inp, cr, cw)
                                if isinstance(v, int)))
    return d


def _csv_text(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


class TestMapping(unittest.TestCase):
    def test_token_split_and_invariant(self):
        rows = [_row("cA", "2026-10-06T12:00:10.000Z", 2.0, 5, 50, 90_000, 700)]
        convs, skipped = build_conversations(rows)
        n = convs["cA"]["nodes"][0]
        self.assertEqual(skipped, 0)
        self.assertEqual(n["cached"], 90_000)        # cacheRead
        self.assertEqual(n["uncached"], 705)         # cacheWrite + input
        self.assertEqual(n["in"], n["cached"] + n["uncached"])

    def test_lane_relative_timing(self):
        rows = [_row("cA", "2026-10-06T12:00:10.000Z", 2.0, 1, 1, 10, 0),
                _row("cA", "2026-10-06T12:01:10.000Z", 5.0, 1, 1, 10, 0)]
        nodes = build_conversations(rows)[0]["cA"]["nodes"]
        self.assertEqual(nodes[0]["startS"], 0.0)    # first call anchors t=0
        self.assertEqual(nodes[0]["endS"], 2.0)
        # t=0 is the FIRST call's start (12:00:10 - 2s = 12:00:08), so a
        # call ending 12:01:10 after 5s of latency starts at +57s
        self.assertEqual(nodes[1]["startS"], 57.0)
        self.assertEqual(nodes[1]["endS"], 62.0)

    def test_rows_sorted_by_start_not_file_order(self):
        rows = [_row("cA", "2026-10-06T12:05:00.000Z", 1.0, 1, 1, 10, 0),
                _row("cA", "2026-10-06T12:00:00.000Z", 1.0, 1, 1, 20, 0)]
        nodes = build_conversations(rows)[0]["cA"]["nodes"]
        self.assertEqual([n["cached"] for n in nodes], [20, 10])

    def test_subagent_lane_nests_under_a_subagent_node(self):
        key = "agent:main:subagent:abc-123"
        rows = [_row("cS", "2026-10-06T12:00:01.000Z", 1.0, 1, 2, 30, 0,
                     kind="subagent", key=key)]
        st = build_conversations(rows)[0]["cS"]
        self.assertEqual(st["nodes"][0]["kind"], "subagent")
        self.assertEqual(st["nodes"][0]["agentId"], "abc-123")
        self.assertEqual(st["totals"]["numSubagentGroups"], 1)
        self.assertEqual(st["totals"]["numTurns"], 0)
        self.assertEqual(len(st["nodes"][0]["children"]), 1)

    def test_blank_usage_columns_are_zero_but_garbage_raises(self):
        r = _row("cA", "2026-10-06T12:00:01.000Z", 1.0, "", 3, "", "")
        self.assertEqual(build_conversations([r])[0]["cA"]["nodes"][0]["in"], 0)
        bad = _row("cA", "2026-10-06T12:00:01.000Z", 1.0, "lots", 3, 0, 0)
        with self.assertRaises(ValueError):
            build_conversations([bad])

    def test_rows_without_timestamp_are_skipped_and_counted(self):
        rows = [_row("cA", "2026-10-06T12:00:01.000Z", 1.0, 1, 1, 10, 0),
                _row("cA", "", 1.0, 1, 1, 10, 0),
                _row("", "2026-10-06T12:00:02.000Z", 1.0, 1, 1, 10, 0)]
        convs, skipped = build_conversations(rows)
        self.assertEqual(skipped, 2)
        self.assertEqual(len(convs["cA"]["nodes"]), 1)

    def test_bad_timestamp_raises_with_row_number(self):
        with self.assertRaises(ValueError) as ctx:
            build_conversations([_row("cA", "last tuesday", 1.0, 1, 1, 1, 0)])
        self.assertIn("row 2", str(ctx.exception))


class TestEndToEnd(unittest.TestCase):
    def _export(self, tmp: Path) -> Path:
        rows = [_row("cA", "2026-10-06T12:00:10.000Z", 2.0, 5, 50, 90_000, 700),
                _row("cA", "2026-10-06T12:02:10.000Z", 3.0, 5, 60, 95_000, 300),
                _row("cS", "2026-10-06T12:01:00.000Z", 1.0, 2, 20, 1_000, 10,
                     kind="subagent", key="agent:main:subagent:zz-9")]
        z = tmp / "export_20261008.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("steps.csv", _csv_text(rows))
            f.writestr("tools.csv", "run_id,tool\n")   # ignored by the import
        return z

    def test_writes_the_standard_cache_and_reloads(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            detail = import_openclaw(self._export(tmp), data_dir=tmp / "data")
            root = tmp / "data" / detail["slug"]
            self.assertTrue(detail["slug"].startswith("openclaw--"))
            self.assertEqual(detail["conversation_count"], 2)
            self.assertEqual(detail["source"], "openclaw")
            self.assertEqual(detail["summary"]["subagentGroups"], 1)
            self.assertEqual(detail["summary"]["mainTurns"], 2)
            index = json.loads(
                (root / "conversations_index.json").read_text(encoding="utf-8"))
            self.assertEqual([i["conv_id"] for i in index], ["cA", "cS"])
            # the records loader accepts these files like AgentX ones
            from modules.records import flatten_conversation
            conv = json.loads(
                (root / "conversations" / "cA.json").read_text(encoding="utf-8"))
            recs = flatten_conversation("cA", conv["structure"])
            self.assertEqual(len(recs), 2)
            self.assertTrue(all(r["in_tokens"] == r["cached_tokens"]
                                + r["uncached_tokens"] for r in recs))
            self.assertEqual({r["role"] for r in recs}, {"main"})

    def test_reimport_replaces_stale_conversations(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            z = self._export(tmp)
            d = import_openclaw(z, data_dir=tmp / "data")
            stale = tmp / "data" / d["slug"] / "conversations" / "ghost.json"
            stale.write_text("{}", encoding="utf-8")
            import_openclaw(z, data_dir=tmp / "data")
            self.assertFalse(stale.exists())

    def test_non_openclaw_csv_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "steps.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_steps(tmp)
            self.assertIn("missing columns", str(ctx.exception))

    def test_missing_steps_csv_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                read_steps(Path(td))


if __name__ == "__main__":
    unittest.main()
