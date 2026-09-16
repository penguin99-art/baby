"""Isolated HTTP fixtures for authenticated API tests."""

import http.client
import json
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import app
from auth_store import AuthStore
from conversation_store import ConversationStore


PASSWORD = "Initial-test-password-123"
CHANGED_PASSWORD = "Changed-test-password-456"
SETUP_TOKEN = "s" * 43


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.temp = self.resources.enter_context(tempfile.TemporaryDirectory())
        self.auth = AuthStore(Path(self.temp) / "accounts.sqlite3", setup_token=SETUP_TOKEN)
        self.store = ConversationStore(Path(self.temp) / "sessions.sqlite3")
        self.resources.callback(self.auth.close)
        self.resources.callback(self.store.close)
        self.resources.enter_context(patch.object(app, "AUTH", self.auth))
        self.resources.enter_context(patch.object(app, "CONVERSATIONS", self.store))
        self.resources.enter_context(patch.object(app, "ensure_store", return_value=[]))
        self.resources.enter_context(patch.object(app, "APP_CONFIG", {}))
        self.admin = self.auth.bootstrap("admin", PASSWORD, SETUP_TOKEN)
        self.cookie = ""
        self.server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        status, _, self.login_cookie_header = self.request("POST", "/api/auth/login", {"username": "admin", "password": PASSWORD})
        self.assertEqual(status, 200)
        self.cookie = self.login_cookie_header.split(";", 1)[0]

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, payload=None, cookie=None, extra=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        headers = {"Content-Type": "application/json", "X-Requested-With": "BabyAssistant"}
        selected_cookie = self.cookie if cookie is None else cookie
        if selected_cookie:
            headers["Cookie"] = selected_cookie
        headers.update(extra or {})
        conn.request(method, path, None if payload is None else json.dumps(payload), headers)
        response = conn.getresponse()
        data = json.loads(response.read())
        result = response.status, data, response.getheader("Set-Cookie")
        conn.close()
        return result

    def create_tester(self, username="tester", *, change_password=True):
        user = self.auth.create_user(self.admin["id"], username, PASSWORD)
        _, _, header = self.request("POST", "/api/auth/login", {"username": username, "password": PASSWORD}, cookie="")
        cookie = header.split(";", 1)[0]
        if change_password:
            status, _, _ = self.request("POST", "/api/auth/password", {"current_password": PASSWORD, "new_password": CHANGED_PASSWORD}, cookie)
            self.assertEqual(status, 200)
            _, _, header = self.request("POST", "/api/auth/login", {"username": username, "password": CHANGED_PASSWORD}, cookie="")
            cookie = header.split(";", 1)[0]
        return user, cookie
