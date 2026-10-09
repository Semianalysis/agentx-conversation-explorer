"""Session source list: privacy default, add/remove, export/restore."""
import unittest

from modules.sources import (LOCAL, SCHEMA, WEBSITE, add, export_blob,
                             make_source, parse_blob, remove, resolve,
                             slugs, source_kind)


class TestSourceEntries(unittest.TestCase):
    def test_kind_comes_from_the_detail_card(self):
        self.assertEqual(source_kind({"slug": "cc-traces"}), WEBSITE)
        self.assertEqual(source_kind({"slug": "x", "source": "openclaw"}), LOCAL)
        self.assertEqual(source_kind({"slug": "x", "source": "hf"}), LOCAL)

    def test_bad_kind_or_missing_slug_raises(self):
        with self.assertRaises(ValueError):
            make_source("s", "sideways", "L")
        with self.assertRaises(ValueError):
            make_source("", WEBSITE, "L")

    def test_label_defaults_to_the_slug(self):
        self.assertEqual(make_source("s", WEBSITE, "")["label"], "s")


class TestListOps(unittest.TestCase):
    def test_add_is_copy_on_write_and_replaces_by_slug(self):
        a = [make_source("x", WEBSITE, "X")]
        b = add(a, make_source("x", LOCAL, "X local", "/p.zip"))
        self.assertEqual(len(a), 1)                   # input untouched
        self.assertEqual(len(b), 1)                   # replaced, not doubled
        self.assertEqual(b[0]["kind"], LOCAL)
        self.assertEqual(b[0]["path"], "/p.zip")

    def test_website_sources_sort_before_local(self):
        out = add(add([], make_source("zz", LOCAL, "zz")),
                  make_source("aa", WEBSITE, "aa"))
        self.assertEqual([s["slug"] for s in out], ["aa", "zz"])

    def test_remove_and_slugs_by_kind(self):
        lst = add(add([], make_source("w", WEBSITE, "w")),
                  make_source("l", LOCAL, "l"))
        self.assertEqual(slugs(lst, LOCAL), ["l"])
        self.assertEqual(slugs(lst, WEBSITE), ["w"])
        self.assertEqual(slugs(remove(lst, "l")), ["w"])


class TestSessionFile(unittest.TestCase):
    def test_roundtrip(self):
        lst = add(add([], make_source("w", WEBSITE, "Published set")),
                  make_source("openclaw--x", LOCAL, "Mine", "C:/e.zip"))
        blob = export_blob(lst)
        self.assertEqual(blob["schema"], SCHEMA)
        self.assertEqual(parse_blob(blob), lst)

    def test_export_carries_no_trace_data(self):
        blob = export_blob([make_source("openclaw--x", LOCAL, "Mine", "e.zip")])
        self.assertEqual(set(blob["sources"][0]),
                         {"slug", "kind", "label", "path"})

    def test_foreign_or_broken_files_raise_with_context(self):
        for bad, msg in (({"schema": "something/else", "version": 1,
                           "sources": []}, "not an explorer session file"),
                         ({"schema": SCHEMA, "version": 99,
                           "sources": []}, "version"),
                         ({"schema": SCHEMA, "version": 1}, "no 'sources'"),
                         ([], "not an object")):
            with self.assertRaises(ValueError) as ctx:
                parse_blob(bad)
            self.assertIn(msg, str(ctx.exception))

    def test_restore_reports_sources_this_machine_lacks(self):
        lst = [make_source("here", LOCAL, "here"),
               make_source("gone", LOCAL, "gone", "D:/elsewhere.zip")]
        present, missing = resolve(lst, {"here"})
        self.assertEqual(slugs(present), ["here"])
        self.assertEqual(slugs(missing), ["gone"])     # reported, not dropped


if __name__ == "__main__":
    unittest.main()
