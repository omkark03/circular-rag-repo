"""Regression tests for store.py and auth.py. Each test redirects
config.DB_PATH to a fresh temp file so tests never interfere with each
other or with a real deployment's database.

Run with:  pytest tests/test_store_and_auth.py -v
Or standalone:  python tests/test_store_and_auth.py
"""
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


@contextmanager
def isolated_db():
    """Redirect config.DB_PATH to a fresh temp file for the duration of a
    test, then clean up. Must re-import store fresh each time since it
    caches nothing module-level except the path itself (read via config
    at call time), so no reload is actually needed — just point elsewhere."""
    tmpdir = tempfile.mkdtemp()
    old_path = config.DB_PATH
    config.DB_PATH = Path(tmpdir) / "test.db"
    import store
    store.init_db()
    try:
        yield store
    finally:
        config.DB_PATH = old_path


# ---------------------------------------------------------------- settings

def test_cross_reference_toggle_defaults_enabled():
    with isolated_db() as store:
        assert store.cross_reference_check_enabled() is True


def test_cross_reference_toggle_persists():
    with isolated_db() as store:
        store.set_cross_reference_check_enabled(False)
        assert store.cross_reference_check_enabled() is False
        store.set_cross_reference_check_enabled(True)
        assert store.cross_reference_check_enabled() is True


def test_settings_upsert_no_duplicate_key_error():
    with isolated_db() as store:
        for _ in range(3):
            store.set_cross_reference_check_enabled(False)
        assert store.cross_reference_check_enabled() is False


def test_stats_missing_refs_respects_toggle():
    with isolated_db() as store:
        a = store.add_document(filename='a.pdf', sha256='hA', status='ok',
                               circular_no='X/1')
        b = store.add_document(filename='b.pdf', sha256='hB', status='ok')
        store.add_reference(b, '99/2099', '99/2099')  # unresolved

        assert store.stats()['missing_refs'] == 1
        store.set_cross_reference_check_enabled(False)
        assert store.stats()['missing_refs'] == 0
        store.set_cross_reference_check_enabled(True)
        assert store.stats()['missing_refs'] == 1


# ---------------------------------------------------------------- documents & references

def test_delete_document_unresolves_citing_references():
    """Deleting a circular that others cite must flip those citations back
    to 'missing' — not leave them silently pointing at a ghost row."""
    with isolated_db() as store:
        citing = store.add_document(filename='citing.pdf', sha256='hC',
                                    status='ok', circular_no='X/1')
        cited = store.add_document(filename='cited.pdf', sha256='hD',
                                   status='ok', circular_no='10/15')
        store.add_reference(citing, '10/15', '10/15')
        assert store.resolve_pending_references() == 1
        assert store.unresolved_references() == []

        store.delete_document(cited)
        assert len(store.unresolved_references()) == 1


def test_bulk_delete_reports_deleted_and_not_found():
    with isolated_db() as store:
        a = store.add_document(filename='a.pdf', sha256='hA', status='ok')
        b = store.add_document(filename='b.pdf', sha256='hB', status='ok')
        c = store.add_document(filename='c.pdf', sha256='hC', status='ok')

        import ingest
        deleted, not_found = [], []
        for doc_id in [a, b, 99999]:
            row = ingest.delete_document(doc_id)
            (deleted if row else not_found).append(doc_id)

        assert set(deleted) == {a, b}
        assert not_found == [99999]
        assert store.get_document(a) is None
        assert store.get_document(c) is not None  # untouched


def test_manual_circular_no_correction_resolves_reference():
    """Manually correcting a document's own number (e.g. after a detection
    failure) must immediately resolve any citations that were waiting on
    it — even if that document's status isn't 'ok' (e.g. ocr_suspect)."""
    with isolated_db() as store:
        citing = store.add_document(filename='citing.pdf', sha256='hC',
                                    status='ok', circular_no='X/1')
        store.add_reference(citing, '10/15', '10/15')
        assert len(store.unresolved_references()) == 1

        suspect = store.add_document(filename='suspect.pdf', sha256='hS',
                                     status='ocr_suspect', circular_no=None)
        store.update_document(suspect, circular_no='10/15')
        fixed = store.resolve_pending_references()
        assert fixed == 1
        assert store.unresolved_references() == []


def test_upload_failed_documents_never_resolve_references():
    """A document that failed to even upload never really existed — it
    must not be usable as a resolution target for citations."""
    with isolated_db() as store:
        citing = store.add_document(filename='citing.pdf', sha256='hC',
                                    status='ok', circular_no='X/1')
        store.add_reference(citing, '99/2024', '99/2024')
        store.add_document(filename='bad.pdf', status='upload_failed',
                          circular_no='99/2024')
        store.resolve_pending_references()
        missing = [row["ref_norm"] for row in store.unresolved_references()]
        assert '99/2024' in missing


# ---------------------------------------------------------------- schema migration

def test_migration_adds_text_sample_column_without_data_loss():
    """A pre-existing database created before the text_sample column
    existed must gain the column on init_db() without losing any data."""
    import sqlite3
    tmpdir = tempfile.mkdtemp()
    old_path = config.DB_PATH
    config.DB_PATH = Path(tmpdir) / "legacy.db"
    try:
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("""CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
            stored_name TEXT, circular_no TEXT, issuer TEXT, title TEXT,
            year TEXT, sha256 TEXT UNIQUE, status TEXT NOT NULL,
            status_detail TEXT, n_chunks INTEGER DEFAULT 0,
            uploaded_at TEXT NOT NULL
        )""")
        conn.execute("INSERT INTO documents (filename, status, uploaded_at) "
                    "VALUES (?,?,?)", ("old_doc.pdf", "ok", "2026-01-01"))
        conn.commit()
        conn.close()

        import store
        store.init_db()
        row = store.get_document(1)
        assert row["filename"] == "old_doc.pdf"
        assert "text_sample" in row.keys()
    finally:
        config.DB_PATH = old_path


# ---------------------------------------------------------------- auth

def test_password_hash_and_verify():
    import auth
    h = auth.hash_password("correct-horse-battery-staple")
    assert auth.verify_password("correct-horse-battery-staple", h)
    assert not auth.verify_password("wrong-password", h)


def test_token_create_and_decode():
    import auth
    token = auth.create_token("alice", "admin")
    payload = auth.decode_token(token)
    assert payload["u"] == "alice"
    assert payload["r"] == "admin"


def test_tampered_token_rejected():
    import auth
    token = auth.create_token("alice", "admin")
    tampered = token[:-2] + "xx"
    assert auth.decode_token(tampered) is None


def test_authenticate_full_flow():
    with isolated_db() as store:
        import auth
        auth.ensure_default_admin()
        user = auth.authenticate("admin", "admin123")
        assert user is not None and user["role"] == "admin"
        assert auth.authenticate("admin", "wrong-password") is None


def test_chat_ownership_isolation():
    """One user's chats must never be visible to another user."""
    with isolated_db() as store:
        import auth
        auth.ensure_default_admin()
        uid_a = store.create_user("alice", auth.hash_password("pw1"), "viewer")
        uid_b = store.create_user("bob", auth.hash_password("pw2"), "viewer")
        chat_a = store.create_chat(uid_a, "alice's chat")
        assert store.chat_owner(chat_a["id"]) == uid_a
        assert store.list_chats(uid_b) == []


def test_chat_history_for_llm_prefers_english_over_translated_content():
    """Regression: once a turn is translated, 'content' holds Marathi text
    shown to the user. Passing that Marathi text back into the chat LLM's
    OWN context on later turns was found to bias subsequent generations
    toward Marathi even when a new question is asked in English. The LLM-
    facing history must always use the English original (content_en) for
    assistant turns, never the displayed/translated content."""
    with isolated_db() as store:
        import auth
        auth.ensure_default_admin()
        admin = store.get_user("admin")
        chat = store.create_chat(admin["id"], "test")
        store.add_message(chat["id"], "user", "पोलीस स्टेशन निरीक्षण कसे होते?")
        store.add_message(
            chat["id"], "assistant",
            "पोलीस अधीक्षकांनी वर्षातून किमान एकदा तपासणी करावी.",  # shown to user
            references=[],
            content_en="The Superintendent of Police must inspect at least once a year.",
        )

        history = store.chat_history_for_llm(chat["id"])
        assert history[1]["content"] == \
            "The Superintendent of Police must inspect at least once a year."
        assert "पोलीस अधीक्षकांनी" not in history[1]["content"]


def test_chat_history_for_llm_falls_back_when_not_translated():
    """A turn that was never translated (content_en is None) must still
    appear in history using its regular content — the fallback must not
    silently drop untranslated messages."""
    with isolated_db() as store:
        import auth
        auth.ensure_default_admin()
        admin = store.get_user("admin")
        chat = store.create_chat(admin["id"], "test")
        store.add_message(chat["id"], "user", "What is the KYC limit?")
        store.add_message(chat["id"], "assistant", "The limit is Rs. 50,000.")
        history = store.chat_history_for_llm(chat["id"])
        assert history[1]["content"] == "The limit is Rs. 50,000."


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
