import os
import sys
import tempfile
import unittest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import config  # noqa: E402
import db.store as store_mod  # noqa: E402
from db.store import SqliteStore  # noqa: E402


class TestSqliteStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = SqliteStore(os.path.join(self.tmp, "t.db"))

    def test_insert_returns_id_and_list_reads_back(self):
        i = self.store.insert("projects", {"filename": "a.mp4", "keyframes": 5})
        self.assertTrue(i)
        rows = self.store.list("projects")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["filename"], "a.mp4")
        self.assertEqual(rows[0]["id"], i)

    def test_count_is_isolated_per_collection(self):
        self.store.insert("a", {"x": 1})
        self.store.insert("b", {"x": 2})
        self.assertEqual(self.store.count("a"), 1)
        self.assertEqual(self.store.count("b"), 1)

    def test_list_is_newest_first(self):
        self.store.insert("c", {"n": 1, "created_at": "2020-01-01T00:00:00.000000"})
        self.store.insert("c", {"n": 2, "created_at": "2020-01-02T00:00:00.000000"})
        rows = self.store.list("c")
        self.assertEqual(rows[0]["n"], 2)

    def test_healthy(self):
        self.assertTrue(self.store.healthy())


class TestStoreFallback(unittest.TestCase):
    def test_supabase_without_keys_falls_back_to_sqlite(self):
        store_mod._store = None
        old = config.settings.USE_SUPABASE
        config.settings.USE_SUPABASE = True
        try:
            s = store_mod.get_store()
            self.assertEqual(s.backend, "sqlite")  # no keys -> graceful fallback, app stays up
        finally:
            config.settings.USE_SUPABASE = old
            store_mod._store = None


if __name__ == "__main__":
    unittest.main()
