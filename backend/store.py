"""SQLite store: documents, status flags, cross-references, and chat sessions."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    stored_name TEXT,
    circular_no TEXT,
    issuer TEXT,
    title TEXT,
    year TEXT,
    sha256 TEXT UNIQUE,
    status TEXT NOT NULL,
    status_detail TEXT,
    n_chunks INTEGER DEFAULT 0,
    text_sample TEXT,               -- first ~500 chars of extracted text, for
                                    -- diagnosing OCR quality regardless of status
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS doc_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ref_text TEXT NOT NULL,
    ref_norm TEXT NOT NULL,
    resolved_doc_id INTEGER,
    UNIQUE(doc_id, ref_norm)
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'viewer',   -- admin | uploader | viewer
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,               -- 'user' | 'assistant'
    content TEXT NOT NULL,
    content_en TEXT,                  -- English original, when content was translated
    references_json TEXT,             -- APA reference objects for assistant turns
    contradictions_json TEXT,         -- detected cross-circular contradictions, if any
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(SCHEMA)
        _migrate(c)


def _migrate(c):
    """Add columns to pre-existing databases that predate this column.
    Safe to run every startup — checks before altering."""
    cols = {row["name"] for row in c.execute("PRAGMA table_info(documents)")}
    if "text_sample" not in cols:
        c.execute("ALTER TABLE documents ADD COLUMN text_sample TEXT")
    msg_cols = {row["name"] for row in c.execute("PRAGMA table_info(messages)")}
    if "content_en" not in msg_cols:
        c.execute("ALTER TABLE messages ADD COLUMN content_en TEXT")
    if "contradictions_json" not in msg_cols:
        c.execute("ALTER TABLE messages ADD COLUMN contradictions_json TEXT")


# ---------------------------------------------------------------- settings

def get_setting(key: str, default: str | None = None) -> str | None:
    with get_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with get_conn() as c:
        c.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))


def cross_reference_check_enabled() -> bool:
    return get_setting("cross_reference_check_enabled", "1") != "0"


def set_cross_reference_check_enabled(enabled: bool):
    set_setting("cross_reference_check_enabled", "1" if enabled else "0")


# ---------------------------------------------------------------- documents

def add_document(**kw) -> int:
    kw.setdefault("uploaded_at", _now())
    cols = ", ".join(kw)
    ph = ", ".join("?" * len(kw))
    with get_conn() as c:
        cur = c.execute(f"INSERT INTO documents ({cols}) VALUES ({ph})", list(kw.values()))
        return cur.lastrowid


def update_document(doc_id: int, **kw):
    sets = ", ".join(f"{k}=?" for k in kw)
    with get_conn() as c:
        c.execute(f"UPDATE documents SET {sets} WHERE id=?", [*kw.values(), doc_id])


def find_by_hash(sha256: str):
    with get_conn() as c:
        return c.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()


def all_documents():
    with get_conn() as c:
        return c.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()


def get_document(doc_id: int):
    with get_conn() as c:
        return c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()


def delete_document(doc_id: int) -> dict | None:
    """Delete a document row. References FROM it are removed (FK cascade);
    references TO it from other circulars become unresolved again so the
    dashboard flags them as missing."""
    with get_conn() as c:
        row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return None
        c.execute("UPDATE doc_references SET resolved_doc_id=NULL WHERE resolved_doc_id=?",
                  (doc_id,))
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        return dict(row)


def failed_documents():
    with get_conn() as c:
        return c.execute(
            "SELECT * FROM documents WHERE status != 'ok' ORDER BY uploaded_at DESC"
        ).fetchall()


# ---------------------------------------------------------------- references

def add_reference(doc_id: int, ref_text: str, ref_norm: str, resolved_doc_id=None):
    with get_conn() as c:
        c.execute(
            """INSERT OR IGNORE INTO doc_references
               (doc_id, ref_text, ref_norm, resolved_doc_id) VALUES (?,?,?,?)""",
            (doc_id, ref_text, ref_norm, resolved_doc_id),
        )


def clear_references(doc_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM doc_references WHERE doc_id=?", (doc_id,))


def unresolved_references():
    with get_conn() as c:
        return c.execute(
            """SELECT r.*, d.filename AS source_file, d.circular_no AS source_circular
               FROM doc_references r JOIN documents d ON d.id = r.doc_id
               WHERE r.resolved_doc_id IS NULL ORDER BY d.filename"""
        ).fetchall()


def resolve_pending_references() -> int:
    """Fuzzy-match unresolved citations against repository circular numbers
    (exact, suffix, or distinctive-containment matches — see references.refs_match).
    Any document with a known circular_no is a valid target, regardless of
    its indexing status — a manually-corrected or ocr_suspect document still
    represents a real circular that other documents can cite."""
    import references as refx
    with get_conn() as c:
        docs = c.execute(
            "SELECT id, circular_no FROM documents "
            "WHERE circular_no IS NOT NULL AND status != 'upload_failed'"
        ).fetchall()
        pending = c.execute(
            "SELECT id, ref_norm, doc_id FROM doc_references WHERE resolved_doc_id IS NULL"
        ).fetchall()
        fixed = 0
        for row in pending:
            for d in docs:
                if d["id"] != row["doc_id"] and refx.refs_match(row["ref_norm"],
                                                                d["circular_no"]):
                    c.execute("UPDATE doc_references SET resolved_doc_id=? WHERE id=?",
                              (d["id"], row["id"]))
                    fixed += 1
                    break
        return fixed


def stats():
    with get_conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        ok = c.execute("SELECT COUNT(*) n FROM documents WHERE status='ok'").fetchone()["n"]
        if cross_reference_check_enabled():
            missing = c.execute(
                "SELECT COUNT(*) n FROM doc_references WHERE resolved_doc_id IS NULL"
            ).fetchone()["n"]
        else:
            missing = 0
        return {"total": total, "ok": ok, "failed": total - ok, "missing_refs": missing}


# ---------------------------------------------------------------- chats

def create_chat(user_id: int, title: str = "New chat") -> dict:
    now = _now()
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO chats (user_id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (user_id, title, now, now),
        )
        return {"id": cur.lastrowid, "user_id": user_id, "title": title,
                "created_at": now, "updated_at": now}


def list_chats(user_id: int):
    with get_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM chats WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,)).fetchall()]


def chat_owner(chat_id: int):
    with get_conn() as c:
        r = c.execute("SELECT user_id FROM chats WHERE id=?", (chat_id,)).fetchone()
        return r["user_id"] if r else None


# ---------------------------------------------------------------- users

def create_user(username: str, password_hash: str, role: str) -> int:
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            (username, password_hash, role, _now()))
        return cur.lastrowid


def get_user(username: str):
    with get_conn() as c:
        return c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()


def list_users():
    with get_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()]


def delete_user(user_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM users WHERE id=?", (user_id,))


def update_user_password(username: str, password_hash: str):
    with get_conn() as c:
        c.execute("UPDATE users SET password_hash=? WHERE username=?",
                  (password_hash, username))


def rename_chat(chat_id: int, title: str):
    with get_conn() as c:
        c.execute("UPDATE chats SET title=?, updated_at=? WHERE id=?",
                  (title[:80], _now(), chat_id))


def delete_chat(chat_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM chats WHERE id=?", (chat_id,))


def touch_chat(chat_id: int):
    with get_conn() as c:
        c.execute("UPDATE chats SET updated_at=? WHERE id=?", (_now(), chat_id))


def add_message(chat_id: int, role: str, content: str, references=None,
                content_en: str | None = None, contradictions=None) -> int:
    with get_conn() as c:
        cur = c.execute(
            """INSERT INTO messages (chat_id, role, content, content_en,
               references_json, contradictions_json, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (chat_id, role, content, content_en,
             json.dumps(references) if references else None,
             json.dumps(contradictions) if contradictions else None, _now()),
        )
        return cur.lastrowid


def chat_messages(chat_id: int):
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY id", (chat_id,)
        ).fetchall()
    out = []
    for r in rows:
        m = dict(r)
        m["references"] = json.loads(m.pop("references_json") or "[]")
        m["contradictions"] = json.loads(m.pop("contradictions_json") or "[]")
        out.append(m)
    return out


def chat_history_for_llm(chat_id: int) -> list[dict]:
    """History formatted for the chat LLM's prompt context — always prefers
    the English original (content_en) over the displayed content for
    assistant turns. This matters: once a turn gets translated, 'content'
    holds the Marathi text shown to the user, and passing THAT back into
    the LLM's own context on later turns was found to bias subsequent
    generations toward Marathi even when the new question is in English —
    the chat LLM should always reason in English internally; translation is
    a separate, downstream display step."""
    return [{"role": m["role"], "content": m.get("content_en") or m["content"]}
           for m in chat_messages(chat_id)]
