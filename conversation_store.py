"""ConversationStore: single-process demo conversation persistence.

Backed by the Python standard library ``sqlite3`` module. No external
dependencies.

Design notes / limitations
--------------------------
* This is a **single-process demo**. Concurrency is serialized with
  in-process ``threading.RLock`` objects (64 stripes keyed by token hash).
  Multi-process access to the same SQLite file is **not** guaranteed to be
  safe or consistent.
* Tokens are never stored in plaintext: only ``sha256(token)`` digests are
  written to the database.
* The database file is created lazily on the first operation, never at
  module import time.
* Sessions have a fixed TTL (``ttl_seconds``, default 86400). The TTL is
  **not** renewed on activity (no sliding expiration).
* ``generate(history)`` receives the most recent 6 messages (the last 3
  turns: user + assistant). Each message carries only ``role`` and
  ``content[:1000]``; assistant messages also carry ``route``.
* The callback result is ``{"content": ..., "route": ...,
  "response": ...}`` where ``response`` wraps the app's existing result
  unchanged; assistant messages persist all three fields for UI recovery.
* Returned result bodies are copies; mutating them does not affect the
  cached/stored conversation state.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from copy import deepcopy

__all__ = ["ConversationStore", "ConversationConflict"]

_NUM_STRIPES = 64
_HISTORY_TURNS = 3
_HISTORY_LIMIT = _HISTORY_TURNS * 2  # 6 messages: 3 turns of user+assistant
_CONTENT_LIMIT = 1000


class ConversationConflict(ValueError):
    """Raised when a request conflicts with the stored conversation state."""


class ConversationStore:
    """Local, single-process conversation store backed by stdlib SQLite.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the SQLite database file. Parent directories are created
        if missing.
    ttl_seconds : float
        Fixed lifetime of a session in seconds (default 86400). Not renewed.

    The ``generate`` callback must return ``{"content", "route",
    "response"}`` where ``response`` wraps the app's existing result
    unchanged; assistant messages persist all three fields for UI recovery.
    """

    def __init__(self, path, ttl_seconds=86400):
        self.path = str(path)
        self.ttl_seconds = ttl_seconds
        self._locks = [threading.RLock() for _ in range(_NUM_STRIPES)]
        self._local = threading.local()

    # ------------------------------------------------------------------
    # connection helpers
    # ------------------------------------------------------------------

    def _connect(self):
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA secure_delete=ON")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " token_hash TEXT NOT NULL UNIQUE,"
            " revision INTEGER NOT NULL DEFAULT 0,"
            " expires_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS turns ("
            " session_id INTEGER NOT NULL"
            "   REFERENCES sessions(id) ON DELETE CASCADE,"
            " request_id TEXT NOT NULL,"
            " expected_revision INTEGER NOT NULL,"
            " query TEXT NOT NULL,"
            " result_json TEXT NOT NULL,"
            " revision INTEGER NOT NULL,"
            " PRIMARY KEY (session_id, request_id))"
        )
        return conn

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def close(self):
        """Close the current thread's database connection, if any."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # token / locking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_token(token):
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _lock_for(self, token):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return self._locks[int.from_bytes(digest[:8], "big") % _NUM_STRIPES]

    # ------------------------------------------------------------------
    # internal data access
    # ------------------------------------------------------------------

    def _cleanup_expired(self):
        # autocommit mode (isolation_level=None): DELETE is committed at once
        self._conn().execute(
            "DELETE FROM sessions WHERE expires_at <= ?", (time.time(),)
        )

    def _get_or_create_session(self, token):
        token_hash = self._hash_token(token)
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        if row is not None:
            return row
        cur = conn.execute(
            "INSERT INTO sessions (token_hash, revision, expires_at)"
            " VALUES (?, 0, ?)",
            (token_hash, time.time() + self.ttl_seconds),
        )
        return conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (cur.lastrowid,)
        ).fetchone()

    def _get_turn(self, session_id, request_id):
        return self._conn().execute(
            "SELECT * FROM turns WHERE session_id = ? AND request_id = ?",
            (session_id, request_id),
        ).fetchone()

    def _build_history(self, session_id):
        """Most recent ``_HISTORY_LIMIT`` messages (the last
        ``_HISTORY_TURNS`` turns: user + assistant) for ``generate``.

        Each message carries only ``role`` and ``content[:1000]``; assistant
        messages also carry ``route``.
        """
        rows = self._conn().execute(
            "SELECT query, result_json FROM turns"
            " WHERE session_id = ? ORDER BY revision DESC LIMIT ?",
            (session_id, _HISTORY_TURNS),
        ).fetchall()
        history = []
        for row in reversed(rows):
            result = json.loads(row["result_json"])
            content = result.get("content", "")
            if isinstance(content, str):
                content = content[:_CONTENT_LIMIT]
            history.append(
                {
                    "role": "user",
                    "content": row["query"][:_CONTENT_LIMIT],
                }
            )
            history.append(
                {
                    "role": "assistant",
                    "content": content,
                    "route": result.get("route"),
                }
            )
        return history

    def _rebuild_messages(self, session_id):
        """Rebuild the full message list from stored turns (by revision)."""
        rows = self._conn().execute(
            "SELECT query, result_json FROM turns"
            " WHERE session_id = ? ORDER BY revision ASC",
            (session_id,),
        ).fetchall()
        messages = []
        for row in rows:
            messages.append({"role": "user", "content": row["query"]})
            assistant = {"role": "assistant"}
            assistant.update(json.loads(row["result_json"]))
            messages.append(assistant)
        return messages

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get(self, token):
        """Return ``{"revision": int, "messages": [dict, ...]}``.

        An empty conversation is created automatically for unknown tokens.
        """
        with self._lock_for(token):
            self._cleanup_expired()
            session = self._get_or_create_session(token)
            return {
                "revision": session["revision"],
                "messages": self._rebuild_messages(session["id"]),
            }

    def clear(self, token):
        """Delete all history for ``token`` and return a fresh empty
        conversation with an incremented revision.

        The revision bump makes requests issued against the old history
        fail with :class:`ConversationConflict` instead of being replayed.
        """
        with self._lock_for(token):
            self._cleanup_expired()
            session = self._get_or_create_session(token)
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM turns WHERE session_id = ?", (session["id"],)
                )
                new_revision = session["revision"] + 1
                conn.execute(
                    "UPDATE sessions SET revision = ? WHERE id = ?",
                    (new_revision, session["id"]),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return {"revision": new_revision, "messages": []}

    def turn(self, token, request_id, expected_revision, query, generate):
        """Run one turn of a conversation.

        ``generate(history)`` receives the most recent 6 messages (the last
        3 turns: user + assistant). Each message carries only ``role`` and
        ``content[:1000]``; assistant messages also carry ``route``. It must
        return a result dict ``{"content": ..., "route": ...,
        "response": ...}`` where ``response`` wraps the app's existing
        result unchanged.

        Returns ``{"result": dict, "revision": int, "request_id": str}``.

        * A duplicate request (same ``request_id``, ``query`` and
          ``expected_revision``) returns the cached result without calling
          ``generate`` again.
        * Reusing ``request_id`` with different ``query``/``expected_revision``
          or supplying a stale ``expected_revision`` raises
          :class:`ConversationConflict`.
        * The user/assistant messages and the result are committed atomically
          only after ``generate`` returns successfully. If ``generate``
          raises, nothing is written.
        * After ``generate`` returns, the session row is re-checked inside
          the commit transaction; if the session was deleted, expired, or
          its revision changed while generating (e.g. another session's
          expiry cleanup ran concurrently), :class:`ConversationConflict`
          is raised and nothing is committed.
        """
        with self._lock_for(token):
            self._cleanup_expired()
            session = self._get_or_create_session(token)
            session_id = session["id"]

            existing = self._get_turn(session_id, request_id)
            if existing is not None:
                if (
                    existing["query"] != query
                    or existing["expected_revision"] != expected_revision
                ):
                    raise ConversationConflict(
                        "request_id %r reused with a different query or "
                        "expected_revision" % (request_id,)
                    )
                return {
                    "result": deepcopy(json.loads(existing["result_json"])),
                    "revision": existing["revision"],
                    "request_id": request_id,
                }

            if expected_revision != session["revision"]:
                raise ConversationConflict(
                    "expected_revision %r does not match current revision %r"
                    % (expected_revision, session["revision"])
                )

            history = self._build_history(session_id)
            result = generate(history)  # may raise; nothing is committed then
            if not isinstance(result, dict):
                raise TypeError("generate() must return a dict")

            new_revision = session["revision"] + 1
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = conn.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if (
                    current is None
                    or current["expires_at"] <= time.time()
                    or current["revision"] != session["revision"]
                ):
                    raise ConversationConflict(
                        "session changed or expired while generating"
                    )
                conn.execute(
                    "INSERT INTO turns (session_id, request_id,"
                    " expected_revision, query, result_json, revision)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        request_id,
                        expected_revision,
                        query,
                        json.dumps(result),
                        new_revision,
                    ),
                )
                conn.execute(
                    "UPDATE sessions SET revision = ? WHERE id = ?",
                    (new_revision, session_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            return {
                "result": deepcopy(result),
                "revision": new_revision,
                "request_id": request_id,
            }