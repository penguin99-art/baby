"""Local account store for the baby assistant.

Accounts, password hashes and login sessions live in a private SQLite file
next to the other data files. The server only accepts localhost traffic, so
this is a lightweight invite-only user system: an administrator creates
accounts, and each account only sees its own conversations.

Security properties:
- Passwords are hashed with PBKDF2-HMAC-SHA256 (600,000 iterations) and a
  random 16-byte salt. scrypt is preferred but unavailable on the macOS
  system Python (LibreSSL), so PBKDF2 keeps the module portable.
- Session tokens are random 256-bit values; only their SHA-256 digest is stored.
- Failed logins are throttled per username and per IP.
- Unknown users, wrong passwords and disabled accounts all return the same
  error so the response does not reveal which accounts exist.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time

USERNAME_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,31}")
PBKDF2_ITERATIONS = 600_000
PBKDF2_DKLEN = 32
SESSION_LIMIT = 10
THROTTLE_WINDOW = 15 * 60
THROTTLE_USERNAME = 5
THROTTLE_IP = 30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    active INTEGER NOT NULL DEFAULT 1,
    must_change_password INTEGER NOT NULL DEFAULT 1,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS login_failures (
    username TEXT NOT NULL,
    ip TEXT NOT NULL,
    attempted_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_failures_username ON login_failures(username, attempted_at);
CREATE INDEX IF NOT EXISTS idx_failures_ip ON login_failures(ip, attempted_at);
"""


class AuthError(Exception):
    def __init__(self, code: str, status: int):
        super().__init__(code)
        self.code = code
        self.status = status


def _iso(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


class AuthStore:
    def __init__(self, path, *, setup_token=None, session_ttl: int = 86400, clock=None):
        self.path = str(path)
        self.setup_token = setup_token
        self.session_ttl = session_ttl
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._connection = None
        self._dummy_hash_cache = None

    # -- connection handling -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            directory = os.path.dirname(self.path) or "."
            os.makedirs(directory, exist_ok=True)
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(_SCHEMA)
            self._connection = connection
        return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=PBKDF2_DKLEN)
        return salt.hex() + "$" + digest.hex()

    @staticmethod
    def _verify_password(password: str, stored: str) -> bool:
        try:
            salt_hex, digest_hex = stored.split("$", 1)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(digest_hex)
        except (ValueError, AttributeError):
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=PBKDF2_DKLEN)
        return hmac.compare_digest(actual, expected)

    def _dummy_hash(self) -> str:
        if self._dummy_hash_cache is None:
            self._dummy_hash_cache = self._hash_password("dummy-password-for-uniform-timing")
        return self._dummy_hash_cache

    @staticmethod
    def _validate_username(username) -> str:
        if not isinstance(username, str):
            raise AuthError("invalid_username", 400)
        normalized = username.strip().lower()
        if not USERNAME_RE.fullmatch(normalized):
            raise AuthError("invalid_username", 400)
        return normalized

    @staticmethod
    def _validate_password(password) -> str:
        if not isinstance(password, str) or not (12 <= len(password) <= 128) or len(password.encode("utf-8")) > 512 or not password.strip():
            raise AuthError("invalid_password", 400)
        return password

    @staticmethod
    def _public_user(row) -> dict:
        return {
            "id": row[0],
            "username": row[1],
            "role": row[2],
            "active": bool(row[3]),
            "must_change_password": bool(row[4]),
            "created_at": row[6],
        }

    @staticmethod
    def _user_row(connection, user_id):
        return connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()

    def _user_for_token(self, connection, token):
        if not token:
            return None
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        row = connection.execute(
            "SELECT u.*, s.created_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=?", (token_hash,)
        ).fetchone()
        if row is None:
            return None
        if self._clock() - row[7] > self.session_ttl:
            connection.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
            connection.commit()
            return None
        return row[:7]

    def _is_initialized(self, connection) -> bool:
        row = connection.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()
        return bool(row and row[0])

    def _check_throttle(self, connection, username, ip, now) -> None:
        cutoff = now - THROTTLE_WINDOW
        username_count = connection.execute("SELECT COUNT(*) FROM login_failures WHERE username=? AND attempted_at>?", (username, cutoff)).fetchone()[0]
        if username_count >= THROTTLE_USERNAME:
            raise AuthError("login_rate_limited", 429)
        ip_count = connection.execute("SELECT COUNT(*) FROM login_failures WHERE ip=? AND attempted_at>?", (ip, cutoff)).fetchone()[0]
        if ip_count >= THROTTLE_IP:
            raise AuthError("login_rate_limited", 429)
        connection.execute("DELETE FROM login_failures WHERE attempted_at<=?", (cutoff,))
        connection.commit()

    def _record_failure(self, connection, username, ip, now) -> None:
        connection.execute("INSERT INTO login_failures (username, ip, attempted_at) VALUES (?, ?, ?)", (username, ip, now))
        connection.commit()

    # -- public API ----------------------------------------------------------

    def is_initialized(self) -> bool:
        with self._lock:
            return self._is_initialized(self._connect())

    def bootstrap(self, username, password, setup_token) -> dict:
        with self._lock:
            connection = self._connect()
            if self._is_initialized(connection):
                raise AuthError("already_initialized", 409)
            if not self.setup_token or not isinstance(setup_token, str) or not hmac.compare_digest(setup_token, self.setup_token):
                raise AuthError("invalid_setup_token", 403)
            username = self._validate_username(username)
            password = self._validate_password(password)
            user_id = secrets.token_hex(16)
            connection.execute(
                "INSERT INTO users (id, username, role, active, must_change_password, password_hash, created_at) VALUES (?, ?, 'admin', 1, 0, ?, ?)",
                (user_id, username, self._hash_password(password), _iso(self._clock())),
            )
            connection.commit()
            self.setup_token = None
            return self._public_user(connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())

    def create_user(self, actor_id, username, password) -> dict:
        with self._lock:
            connection = self._connect()
            actor = self._user_row(connection, actor_id)
            if not actor or actor[2] != "admin":
                raise AuthError("forbidden", 403)
            username = self._validate_username(username)
            password = self._validate_password(password)
            if connection.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone():
                raise AuthError("username_taken", 409)
            user_id = secrets.token_hex(16)
            connection.execute(
                "INSERT INTO users (id, username, role, active, must_change_password, password_hash, created_at) VALUES (?, ?, 'user', 1, 1, ?, ?)",
                (user_id, username, self._hash_password(password), _iso(self._clock())),
            )
            connection.commit()
            return self._public_user(connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())

    def login(self, username, password, ip) -> tuple[str, dict]:
        with self._lock:
            connection = self._connect()
            now = self._clock()
            normalized = username.strip().lower() if isinstance(username, str) else ""
            self._check_throttle(connection, normalized, ip, now)
            row = connection.execute("SELECT * FROM users WHERE username=?", (normalized,)).fetchone() if USERNAME_RE.fullmatch(normalized) else None
            if row is None:
                self._verify_password(password, self._dummy_hash())
                self._record_failure(connection, normalized, ip, now)
                raise AuthError("invalid_credentials", 401)
            if not self._verify_password(password, row[5]) or not row[3]:
                self._record_failure(connection, normalized, ip, now)
                raise AuthError("invalid_credentials", 401)
            connection.execute("DELETE FROM login_failures WHERE username=? OR ip=?", (normalized, ip))
            connection.execute("DELETE FROM sessions WHERE user_id=? AND created_at<=?", (row[0], now - self.session_ttl))
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
            connection.execute("INSERT INTO sessions (token_hash, user_id, created_at) VALUES (?, ?, ?)", (token_hash, row[0], now))
            connection.execute(
                "DELETE FROM sessions WHERE user_id=? AND token_hash NOT IN (SELECT token_hash FROM sessions WHERE user_id=? ORDER BY created_at DESC LIMIT ?)",
                (row[0], row[0], SESSION_LIMIT),
            )
            connection.commit()
            return token, self._public_user(row)

    def authenticate(self, token) -> dict | None:
        if not token:
            return None
        with self._lock:
            connection = self._connect()
            row = self._user_for_token(connection, token)
            if row is None or not row[3]:
                if row is not None:
                    connection.execute("DELETE FROM sessions WHERE user_id=?", (row[0],))
                    connection.commit()
                return None
            return self._public_user(row)

    def logout(self, token) -> None:
        if not token:
            return
        with self._lock:
            connection = self._connect()
            token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
            connection.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
            connection.commit()

    def change_password(self, token, current_password, new_password) -> None:
        with self._lock:
            connection = self._connect()
            user = self._user_for_token(connection, token)
            if user is None:
                raise AuthError("authentication_required", 401)
            if not self._verify_password(current_password, user[5]):
                raise AuthError("invalid_credentials", 401)
            new_password = self._validate_password(new_password)
            connection.execute("UPDATE users SET password_hash=?, must_change_password=0 WHERE id=?", (self._hash_password(new_password), user[0]))
            connection.execute("DELETE FROM sessions WHERE user_id=?", (user[0],))
            connection.commit()

    def reset_password(self, actor_id, user_id, password) -> dict:
        with self._lock:
            connection = self._connect()
            actor = self._user_row(connection, actor_id)
            if not actor or actor[2] != "admin":
                raise AuthError("forbidden", 403)
            target = self._user_row(connection, user_id)
            if target is None:
                raise AuthError("user_not_found", 404)
            if target[2] == "admin":
                raise AuthError("forbidden", 403)
            password = self._validate_password(password)
            connection.execute("UPDATE users SET password_hash=?, must_change_password=1 WHERE id=?", (self._hash_password(password), user_id))
            connection.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            connection.commit()
            return self._public_user(connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())

    def set_active(self, actor_id, user_id, active) -> dict:
        with self._lock:
            connection = self._connect()
            actor = self._user_row(connection, actor_id)
            if not actor or actor[2] != "admin":
                raise AuthError("forbidden", 403)
            target = self._user_row(connection, user_id)
            if target is None:
                raise AuthError("user_not_found", 404)
            if target[2] == "admin":
                raise AuthError("forbidden", 403)
            connection.execute("UPDATE users SET active=? WHERE id=?", (1 if active else 0, user_id))
            if not active:
                connection.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            connection.commit()
            return self._public_user(connection.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())

    def list_users(self, actor_id) -> list[dict]:
        with self._lock:
            connection = self._connect()
            actor = self._user_row(connection, actor_id)
            if not actor or actor[2] != "admin":
                raise AuthError("forbidden", 403)
            rows = connection.execute("SELECT * FROM users ORDER BY created_at, username").fetchall()
            return [self._public_user(row) for row in rows]