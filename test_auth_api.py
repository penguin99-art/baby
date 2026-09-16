import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from auth_store import AuthStore
from test_support import ApiTestCase, PASSWORD, CHANGED_PASSWORD, SETUP_TOKEN


class AuthApiTests(ApiTestCase):
    def test_unauthenticated_cannot_access_business_apis(self):
        for path in ("/api/session", "/api/docs", "/api/config", "/api/admin/users"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path, cookie="")[0], 401)
        for path in ("/api/ask", "/api/session/clear", "/api/config", "/api/config/test", "/api/upload", "/api/reset", "/api/admin/users"):
            with self.subTest(path=path):
                self.assertEqual(self.request("POST", path, {}, cookie="")[0], 401)

    def test_me_is_public_but_contains_no_credentials(self):
        status, result, _ = self.request("GET", "/api/auth/me", cookie="")
        self.assertEqual(status, 200)
        self.assertIsNone(result["user"])
        self.assertFalse(result["setup_required"])
        self.assertFalse(result["bootstrap_available"])
        _, result, _ = self.request("GET", "/api/auth/me")
        self.assertEqual(result["user"]["role"], "admin")
        serialized = json.dumps(result)
        for secret in (PASSWORD, SETUP_TOKEN, "password_hash", "token_hash", "salt"):
            self.assertNotIn(secret, serialized)

    def test_all_api_gets_require_application_header(self):
        for path in ("/api/auth/me", "/api/session", "/api/docs", "/api/admin/users", "/api/config"):
            self.assertEqual(self.request("GET", path, extra={"X-Requested-With": ""})[0], 403)

    def test_bootstrap_closed_after_first_admin(self):
        status, _, _ = self.request("POST", "/api/auth/bootstrap", {"setup_token": SETUP_TOKEN, "username": "otheradmin", "password": PASSWORD}, cookie="")
        self.assertIn(status, (403, 409))
        self.assertEqual(len(self.auth.list_users(self.admin["id"])), 1)

    def test_bootstrap_needs_private_local_capability(self):
        auth = AuthStore(Path(self.temp) / "fresh.sqlite3", setup_token=SETUP_TOKEN)
        with patch.object(app, "AUTH", auth):
            try:
                status, result, _ = self.request("GET", "/api/auth/me", cookie="")
                self.assertEqual(status, 200)
                self.assertTrue(result["setup_required"])
                self.assertNotIn(SETUP_TOKEN, json.dumps(result))
                payload = {"setup_token": "wrong", "username": "admin", "password": PASSWORD}
                self.assertEqual(self.request("POST", "/api/auth/bootstrap", payload, cookie="")[0], 403)
                payload["setup_token"] = SETUP_TOKEN
                self.assertEqual(self.request("POST", "/api/auth/bootstrap", payload, cookie="")[0], 201)
                status, result, header = self.request("POST", "/api/auth/login", {"username": "admin", "password": PASSWORD}, cookie="")
                self.assertEqual(status, 200)
                self.assertEqual(result["user"]["role"], "admin")
                self.assertIn("HttpOnly", header)
            finally:
                auth.close()

    def test_admin_creates_only_tester_and_no_password_returned(self):
        payload = {"username": "tester", "password": PASSWORD}
        status, result, _ = self.request("POST", "/api/admin/users", payload)
        self.assertEqual(status, 201)
        self.assertEqual(result["user"]["role"], "user")
        self.assertTrue(result["user"]["must_change_password"])
        self.assertNotIn(PASSWORD, json.dumps(result))
        self.assertEqual(self.request("POST", "/api/admin/users", {**payload, "username": "elevated", "role": "admin"})[0], 400)
        self.assertEqual(self.request("POST", "/api/admin/users", payload)[0], 409)

    def test_initial_password_must_change_before_using_app(self):
        user, cookie = self.create_tester(change_password=False)
        status, result, _ = self.request("GET", "/api/auth/me", cookie=cookie)
        self.assertTrue(result["user"]["must_change_password"])
        for method, path in (("GET", "/api/session"), ("GET", "/api/docs"), ("POST", "/api/ask")):
            status, result, _ = self.request(method, path, {} if method == "POST" else None, cookie)
            self.assertEqual(status, 403)
            self.assertEqual(result["error"], "password_change_required")
        status, _, header = self.request("POST", "/api/auth/password", {"current_password": PASSWORD, "new_password": CHANGED_PASSWORD}, cookie)
        self.assertEqual(status, 200)
        self.assertIn("Max-Age=0", header)
        self.assertEqual(self.request("GET", "/api/session", cookie=cookie)[0], 401)
        status, result, _ = self.request("POST", "/api/auth/login", {"username": user["username"], "password": CHANGED_PASSWORD}, cookie="")
        self.assertEqual(status, 200)
        self.assertFalse(result["user"]["must_change_password"])

    def test_tester_cannot_manage_users_config_or_knowledge(self):
        user, cookie = self.create_tester()
        with patch.object(app, "save_store") as save, patch.object(app, "call_llm") as llm:
            for path in ("/api/config", "/api/admin/users"):
                self.assertEqual(self.request("GET", path, cookie=cookie)[0], 403)
            for path in ("/api/config", "/api/config/test", "/api/upload", "/api/reset", "/api/admin/users", "/api/admin/users/status", "/api/admin/users/password"):
                self.assertEqual(self.request("POST", path, {"user_id": user["id"], "active": False}, cookie)[0], 403)
            save.assert_not_called()
            llm.assert_not_called()
        self.assertEqual(self.request("GET", "/api/docs", cookie=cookie)[0], 200)
        self.assertEqual(self.request("POST", "/api/ask", {"query": "你是谁", "request_id": "identity_1", "revision": 0}, cookie)[0], 200)

    def test_login_rotates_old_cookie_and_logout_revokes_session(self):
        status, _, header = self.request("POST", "/api/auth/login", {"username": "admin", "password": PASSWORD})
        self.assertEqual(status, 200)
        cookie = header.split(";", 1)[0]
        self.assertNotEqual(cookie, self.cookie)
        self.assertEqual(self.request("GET", "/api/session")[0], 401)
        self.assertEqual(self.request("GET", "/api/session", cookie=cookie)[0], 200)
        status, _, expired = self.request("POST", "/api/auth/logout", {}, cookie)
        self.assertEqual(status, 200)
        self.assertIn("Max-Age=0", expired)
        self.assertEqual(self.request("GET", "/api/session", cookie=cookie)[0], 401)

    def test_disable_revokes_existing_cookie_and_reenable_does_not_restore_it(self):
        user, cookie = self.create_tester()
        self.assertEqual(self.request("POST", "/api/admin/users/status", {"user_id": user["id"], "active": False})[0], 200)
        self.assertEqual(self.request("GET", "/api/session", cookie=cookie)[0], 401)
        self.assertEqual(self.request("POST", "/api/auth/login", {"username": user["username"], "password": CHANGED_PASSWORD}, cookie="")[0], 401)
        self.assertEqual(self.request("POST", "/api/admin/users/status", {"user_id": user["id"], "active": True})[0], 200)
        self.assertEqual(self.request("GET", "/api/session", cookie=cookie)[0], 401)
        self.assertEqual(self.request("POST", "/api/auth/login", {"username": user["username"], "password": CHANGED_PASSWORD}, cookie="")[0], 200)

    def test_admin_reset_password_forces_change_and_revokes_all_logins(self):
        user, cookie = self.create_tester()
        _, _, header = self.request("POST", "/api/auth/login", {"username": user["username"], "password": CHANGED_PASSWORD}, cookie="")
        second = header.split(";", 1)[0]
        status, result, _ = self.request("POST", "/api/admin/users/password", {"user_id": user["id"], "password": PASSWORD})
        self.assertEqual(status, 200)
        self.assertTrue(result["user"]["must_change_password"])
        for old in (cookie, second):
            self.assertEqual(self.request("GET", "/api/session", cookie=old)[0], 401)
        self.assertEqual(self.request("POST", "/api/auth/login", {"username": user["username"], "password": CHANGED_PASSWORD}, cookie="")[0], 401)

    def test_admin_is_protected_from_disable_and_reset(self):
        for path, payload in (("/api/admin/users/status", {"user_id": self.admin["id"], "active": False}), ("/api/admin/users/password", {"user_id": self.admin["id"], "password": CHANGED_PASSWORD})):
            self.assertIn(self.request("POST", path, payload)[0], (400, 403))
        self.assertEqual(self.request("GET", "/api/session")[0], 200)

    def test_server_owns_conversation_and_old_anonymous_cookie_cannot_claim_it(self):
        legacy = "a" * 43
        self.store.turn(legacy, "old", 0, "private anonymous text", lambda history: {"content": "old", "route": "companion"})
        status, result, _ = self.request("GET", "/api/session", cookie=f"baby_session={legacy}")
        self.assertEqual(status, 401)
        user, cookie = self.create_tester()
        payload = {"query": "你是谁", "request_id": "identity_1", "revision": 0}
        self.assertEqual(self.request("POST", "/api/ask", {**payload, "user_id": self.admin["id"]}, cookie)[0], 400)
        self.assertEqual(self.request("POST", "/api/ask", payload, cookie)[0], 200)
        self.assertEqual(self.request("GET", "/api/session")[1]["messages"], [])
        status, result, _ = self.request("GET", "/api/session", cookie=cookie)
        self.assertEqual(len(result["messages"]), 2)
        self.assertNotIn("private anonymous text", json.dumps(result))

    def test_same_account_recovers_history_after_logout_login(self):
        self.request("POST", "/api/ask", {"query": "先别给建议", "request_id": "request_1", "revision": 0})
        self.request("POST", "/api/auth/logout", {})
        _, _, header = self.request("POST", "/api/auth/login", {"username": "admin", "password": PASSWORD}, cookie="")
        _, state, _ = self.request("GET", "/api/session", cookie=header.split(";", 1)[0])
        self.assertEqual(state["revision"], 1)
        self.assertEqual(state["messages"][-1]["response"]["conversation_state"]["advice_preference"]["mode"], "deferred")

    def test_revocation_during_generation_does_not_save_late_response(self):
        user, cookie = self.create_tester()
        entered, resume = threading.Event(), threading.Event()
        result = []
        def slow(*args, **kwargs):
            entered.set()
            resume.wait(5)
            return {"answer": "late", "route": "companion", "citations": [], "status": "supported"}
        with patch.object(app, "answer", side_effect=slow):
            worker = threading.Thread(target=lambda: result.append(self.request("POST", "/api/ask", {"query": "有些困扰", "request_id": "slow_query_1", "revision": 0}, cookie)))
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(self.request("POST", "/api/admin/users/status", {"user_id": user["id"], "active": False})[0], 200)
            finally:
                resume.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result[0][0], 401)
        self.assertEqual(self.store.get(app.conversation_key(user))["messages"], [])

    def test_auth_rejects_cross_origin_and_unexpected_fields(self):
        payload = {"username": "admin", "password": PASSWORD}
        for headers in ({"Origin": "https://evil.example"}, {"X-Requested-With": ""}, {"Host": "evil.example"}):
            self.assertEqual(self.request("POST", "/api/auth/login", payload, cookie="", extra=headers)[0], 403)
        for payload in ([], None, {"username": "admin", "password": PASSWORD, "role": "admin"}):
            self.assertEqual(self.request("POST", "/api/auth/login", payload, cookie="")[0], 400)

    def test_unknown_and_bad_password_have_same_error(self):
        unknown = self.request("POST", "/api/auth/login", {"username": "unknown", "password": PASSWORD}, cookie="")
        wrong = self.request("POST", "/api/auth/login", {"username": "admin", "password": "incorrect-password"}, cookie="")
        self.assertEqual(unknown[:2], wrong[:2])
        self.assertEqual(wrong[0], 401)


class BootstrapFileTests(unittest.TestCase):
    def test_local_token_is_private_stable_and_not_precreated_admin(self):
        with tempfile.TemporaryDirectory() as directory:
            auth = AuthStore(Path(directory) / "accounts.sqlite3")
            try:
                token_path = app.prepare_bootstrap(auth)
                self.assertEqual(token_path.stat().st_mode & 0o777, 0o600)
                first = auth.setup_token
                self.assertEqual(len(first), 43)
                self.assertFalse(auth.is_initialized())
                self.assertEqual(app.prepare_bootstrap(auth), token_path)
                self.assertEqual(first, auth.setup_token)
                auth.bootstrap("admin", PASSWORD, first)
                self.assertIsNone(app.prepare_bootstrap(auth))
            finally:
                auth.close()


if __name__ == "__main__":
    unittest.main()
