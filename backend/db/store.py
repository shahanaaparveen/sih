"""
Document store for projects / telemetry / pipeline jobs (ROADMAP.md P1.4 - P1.5).

SQLite by default; Supabase when USE_SUPABASE=true and keys are set. The interface is identical
either way, so main.py never cares which backend is live. Any Supabase failure (e.g. a network
drop) falls back to SQLite so the app stays up.

Interface:
    store = get_store()
    store.insert(collection, doc) -> id
    store.list(collection, limit) -> [doc, ...]   # newest first
    store.count(collection) -> int
    store.healthy() -> bool
    store.backend -> "sqlite" | "supabase"
"""
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from config import settings

_ISO = "%Y-%m-%dT%H:%M:%S.%f"


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_ISO)


def _doc_id_created(doc: Dict[str, Any]):
    doc = dict(doc)
    doc_id = str(doc.get("id") or uuid.uuid4())
    created = doc.get("created_at")
    if isinstance(created, datetime):
        created = created.strftime(_ISO)
    created = created or _now()
    doc["created_at"] = created
    doc.pop("id", None)
    return doc_id, created, doc


class SqliteStore:
    backend = "sqlite"

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        with self._lock, self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS documents ("
                      "id TEXT PRIMARY KEY, collection TEXT NOT NULL, "
                      "created_at TEXT NOT NULL, data TEXT NOT NULL)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_docs_coll "
                      "ON documents(collection, created_at DESC)")

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def insert(self, collection: str, doc: Dict[str, Any]) -> str:
        doc_id, created, doc = _doc_id_created(doc)
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?)",
                      (doc_id, collection, created, json.dumps(doc, default=str)))
        return doc_id

    def list(self, collection: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT id, data FROM documents WHERE collection=? "
                             "ORDER BY created_at DESC LIMIT ?", (collection, limit)).fetchall()
        out = []
        for r in rows:
            d = json.loads(r["data"])
            d["id"] = r["id"]
            out.append(d)
        return out

    def count(self, collection: str) -> int:
        with self._lock, self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM documents WHERE collection=?",
                                 (collection,)).fetchone()[0])

    def healthy(self) -> bool:
        try:
            with self._lock, self._conn() as c:
                c.execute("SELECT 1")
            return True
        except Exception:
            return False


class SupabaseStore:
    """Uses a single namespaced `aero3d_documents` table (id, collection, created_at, data jsonb).
    Namespaced so it never collides with other apps sharing the same Supabase project.
    The backend connects with the service_role key (server-side only), so RLS is bypassed;
    anon/authenticated are denied by RLS as defense-in-depth. Kept API-compatible with SqliteStore."""
    backend = "supabase"
    TABLE = "aero3d_documents"

    def __init__(self):
        from supabase import create_client  # lazy import; only when actually used
        if not (settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY):
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set")
        self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        self.client.table(self.TABLE).select("id").limit(1).execute()  # fail fast

    def insert(self, collection: str, doc: Dict[str, Any]) -> str:
        doc_id, created, doc = _doc_id_created(doc)
        row = {"id": doc_id, "collection": collection, "created_at": created,
               "data": json.loads(json.dumps(doc, default=str))}
        self.client.table(self.TABLE).insert(row).execute()
        return doc_id

    def list(self, collection: str, limit: int = 20) -> List[Dict[str, Any]]:
        res = (self.client.table(self.TABLE).select("id,data").eq("collection", collection)
               .order("created_at", desc=True).limit(limit).execute())
        out = []
        for r in res.data or []:
            d = dict(r.get("data") or {})
            d["id"] = r["id"]
            out.append(d)
        return out

    def count(self, collection: str) -> int:
        res = (self.client.table(self.TABLE).select("id", count="exact")
               .eq("collection", collection).execute())
        return int(res.count or 0)

    def healthy(self) -> bool:
        try:
            self.client.table(self.TABLE).select("id").limit(1).execute()
            return True
        except Exception:
            return False


_store = None
_store_lock = threading.Lock()


def get_store():
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        if settings.USE_SUPABASE:
            try:
                _store = SupabaseStore()
                print("[DB] Supabase document store connected")
                return _store
            except Exception as e:
                print(f"[DB] Supabase unavailable ({e}); falling back to SQLite")
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _store = SqliteStore(os.path.join(backend_dir, "storage", "aero3d.db"))
        print(f"[DB] SQLite document store at {_store.path}")
        return _store
