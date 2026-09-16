"""Tests for ConversationStore (stdlib SQLite conversation persistence)."""

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest

from conversation_store import ConversationStore, ConversationConflict


def make_result(content, route="default", response=None, state=None):
    result = {
        "content": content,
        "route": route,
        "response": response if response is not None else {"text": content},
    }
    if state is not None:
        result["state"] = state
    return result


class ConversationStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "conv.db")
        self._stores = []

    def _store(self, **kwargs):
        store = ConversationStore(self.path, **kwargs)
        self._stores.append(store)
        return store

    def tearDown(self):
        for store in self._stores:
            store.close()
        self._tmp.cleanup()

    # ------------------------------------------------------------------

    def test_db_not_created_at_import(self):
        self.assertFalse(os.path.exists(self.path))
        store = self._store()
        self.assertFalse(os.path.exists(self.path))  # lazy: no DB yet
        store.get("tok")
        self.assertTrue(os.path.exists(self.path))

    def test_empty_conversation_auto_created(self):
        store = self._store()
        self.assertEqual(store.get("new-token"), {"revision": 0, "messages": []})

    def test_isolation(self):
        store = self._store()
        store.turn("alice", "r1", 0, "hello", lambda h: make_result("hi alice"))
        store.turn("bob", "r1", 0, "hello", lambda h: make_result("hi bob"))
        alice = store.get("alice")
        bob = store.get("bob")
        self.assertEqual(alice["revision"], 1)
        self.assertEqual([m["content"] for m in alice["messages"]], ["hello", "hi alice"])
        self.assertEqual([m["content"] for m in bob["messages"]], ["hello", "hi bob"])

    def test_restart_persistence(self):
        store = self._store()
        store.turn("tok", "r1", 0, "q1", lambda h: make_result("a1"))
        store.close()
        store2 = self._store()
        data = store2.get("tok")
        self.assertEqual(data["revision"], 1)
        self.assertEqual([m["content"] for m in data["messages"]], ["q1", "a1"])

    def test_duplicate_request(self):
        store = self._store()
        calls = []

        def gen(h):
            calls.append(1)
            return make_result("answer")

        first = store.turn("tok", "req-1", 0, "q", gen)
        second = store.turn("tok", "req-1", 0, "q", gen)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["revision"], 1)
        self.assertEqual(second["revision"], 1)
        self.assertEqual(first["request_id"], "req-1")
        self.assertEqual(second["request_id"], "req-1")
        self.assertEqual(first["result"], second["result"])

    def test_reuse_id_different_content(self):
        store = self._store()
        store.turn("tok", "req-1", 0, "q1", lambda h: make_result("a1"))
        with self.assertRaises(ConversationConflict):
            store.turn("tok", "req-1", 0, "q2", lambda h: make_result("a2"))
        with self.assertRaises(ConversationConflict):
            store.turn("tok", "req-1", 1, "q1", lambda h: make_result("a2"))

    def test_revision_conflict(self):
        store = self._store()
        store.turn("tok", "r1", 0, "q1", lambda h: make_result("a1"))
        with self.assertRaises(ConversationConflict):
            store.turn("tok", "r2", 0, "q2", lambda h: make_result("a2"))

    def test_clear_then_old_request(self):
        store = self._store()
        store.turn("tok", "r1", 0, "q1", lambda h: make_result("a1"))
        cleared = store.clear("tok")
        self.assertEqual(cleared, {"revision": 2, "messages": []})
        # old revision is rejected after clear (no replay)
        with self.assertRaises(ConversationConflict):
            store.turn("tok", "r2", 1, "q2", lambda h: make_result("a2"))
        # current revision works
        ok = store.turn("tok", "r3", 2, "q3", lambda h: make_result("a3"))
        self.assertEqual(ok["revision"], 3)

    def test_callback_failure_no_write(self):
        store = self._store()

        def bad_gen(h):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            store.turn("tok", "r1", 0, "q1", bad_gen)
        data = store.get("tok")
        self.assertEqual(data["revision"], 0)
        self.assertEqual(data["messages"], [])
        # retry with the same request_id/revision succeeds
        ok = store.turn("tok", "r1", 0, "q1", lambda h: make_result("a1"))
        self.assertEqual(ok["revision"], 1)

    def test_concurrent_duplicate_generate_once(self):
        store = self._store()
        calls = []
        calls_lock = threading.Lock()

        def gen(h):
            with calls_lock:
                calls.append(1)
            time.sleep(0.05)
            return make_result("answer")

        results = []
        errors = []

        def worker():
            try:
                results.append(store.turn("tok", "req-1", 0, "q", gen))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["revision"], 1)
        self.assertEqual(results[1]["revision"], 1)

    def test_token_not_stored_plaintext(self):
        store = self._store()
        token = "super-secret-token-xyz"
        store.turn(token, "r1", 0, "q", lambda h: make_result("a"))
        store.close()

        conn = sqlite3.connect(self.path)
        try:
            sessions = conn.execute("SELECT * FROM sessions").fetchall()
            self.assertEqual(len(sessions), 1)
            token_hash = sessions[0][1]
            self.assertRegex(token_hash, r"^[0-9a-f]{64}$")
            self.assertNotEqual(token_hash, token)
            for row in sessions:
                self.assertNotIn(token, str(row))
            turns = conn.execute("SELECT * FROM turns").fetchall()
            self.assertEqual(len(turns), 1)
            for row in turns:
                self.assertNotIn(token, str(row))
        finally:
            conn.close()

    def test_expiry(self):
        store = self._store(ttl_seconds=0.1)
        store.turn("tok", "r1", 0, "q1", lambda h: make_result("a1"))
        time.sleep(0.3)
        data = store.get("tok")
        self.assertEqual(data["revision"], 0)
        self.assertEqual(data["messages"], [])

    def test_fixed_ttl_no_sliding(self):
        store = self._store(ttl_seconds=0.2)
        store.turn("tok", "r1", 0, "q1", lambda h: make_result("a1"))
        time.sleep(0.1)
        store.turn("tok", "r2", 1, "q2", lambda h: make_result("a2"))
        time.sleep(0.15)  # total 0.25 > 0.2: session must be gone
        data = store.get("tok")
        self.assertEqual(data["revision"], 0)
        self.assertEqual(data["messages"], [])

    def test_history_passed_to_generate(self):
        store = self._store()
        captured = {}
        store.turn("tok", "r1", 0, "q1", lambda h: make_result("a1", route="r1"))

        def gen2(h):
            captured["history"] = h
            return make_result("a2")

        store.turn("tok", "r2", 1, "q2", gen2)
        hist = captured["history"]
        self.assertEqual(len(hist), 2)  # 1 turn = user + assistant
        self.assertEqual(hist[0], {"role": "user", "content": "q1"})
        self.assertEqual(hist[1], {"role": "assistant", "content": "a1", "route": "r1"})

    def test_history_limit_six_and_truncation(self):
        store = self._store()
        captured = {}
        for i in range(8):
            store.turn(
                "tok", "r%d" % i, i, "q%d" % i,
                lambda h, i=i: make_result("x" * 1200, route="route%d" % i),
            )

        def gen(h):
            captured["history"] = h
            return make_result("final")

        store.turn("tok", "r8", 8, "q8", gen)
        hist = captured["history"]
        self.assertEqual(len(hist), 6)  # last 3 turns = 6 messages
        self.assertEqual(hist[0], {"role": "user", "content": "q5"})
        self.assertEqual(hist[1], {"role": "assistant", "content": "x" * 1000, "route": "route5"})
        self.assertEqual(hist[2], {"role": "user", "content": "q6"})
        self.assertEqual(hist[3], {"role": "assistant", "content": "x" * 1000, "route": "route6"})
        self.assertEqual(hist[4], {"role": "user", "content": "q7"})
        self.assertEqual(hist[5], {"role": "assistant", "content": "x" * 1000, "route": "route7"})
        self.assertNotIn("response", hist[1])

    def test_history_includes_user_statements(self):
        store = self._store()
        captured = {}
        store.turn("tok", "r1", 0, "user says hello", lambda h: make_result("a1"))
        store.turn("tok", "r2", 1, "user says again", lambda h: make_result("a2"))

        def gen(h):
            captured["history"] = h
            return make_result("a3")

        store.turn("tok", "r3", 2, "user says third", gen)
        hist = captured["history"]
        self.assertEqual(len(hist), 4)  # 2 turns = 4 messages
        self.assertEqual(hist[0], {"role": "user", "content": "user says hello"})
        self.assertEqual(hist[1], {"role": "assistant", "content": "a1", "route": "default"})
        self.assertEqual(hist[2], {"role": "user", "content": "user says again"})
        self.assertEqual(hist[3], {"role": "assistant", "content": "a2", "route": "default"})

    def test_result_wraps_app_result_unchanged(self):
        store = self._store()
        app_result = {"body": "hello world", "meta": {"x": 1}}

        def gen(h):
            return {"content": "Hello!", "route": "chat", "response": app_result}

        ret = store.turn("tok", "r1", 0, "q1", gen)
        self.assertEqual(ret["result"]["response"], app_result)
        # mutating the returned body must not affect the stored copy
        ret["result"]["response"]["body"] = "mutated"
        data = store.get("tok")
        self.assertEqual(data["messages"][1]["response"]["body"], "hello world")

    def test_session_expired_during_generate_conflicts(self):
        store = self._store(ttl_seconds=0.1)
        import hashlib

        def stripe(t):
            d = hashlib.sha256(t.encode("utf-8")).digest()
            return int.from_bytes(d[:8], "big") % 64

        s0 = stripe("tok")
        other = "other"
        while stripe(other) == s0:
            other += "x"

        started = threading.Event()
        outcomes = []

        def slow_gen(h):
            started.set()
            time.sleep(0.3)
            return make_result("late")

        def worker():
            try:
                store.turn("tok", "r1", 0, "q1", slow_gen)
                outcomes.append("committed")
            except ConversationConflict:
                outcomes.append("conflict")
            except Exception:
                outcomes.append("other")

        t = threading.Thread(target=worker)
        t.start()
        started.wait()
        time.sleep(0.15)  # session TTL (0.1s) elapses while generating
        store.get(other)  # another session's cleanup deletes the expired one
        t.join()

        self.assertEqual(outcomes, ["conflict"])
        data = store.get("tok")
        self.assertEqual(data["revision"], 0)
        self.assertEqual(data["messages"], [])

    def test_return_body_copy(self):
        store = self._store()
        result = make_result("a1")
        ret = store.turn("tok", "r1", 0, "q1", lambda h: result)
        ret["result"]["content"] = "mutated"
        cached = store.turn("tok", "r1", 0, "q1", lambda h: make_result("never"))
        self.assertEqual(cached["result"]["content"], "a1")

    def test_messages_include_response_for_ui(self):
        store = self._store()
        store.turn(
            "tok", "r1", 0, "q1",
            lambda h: make_result("a1", route="r", response={"body": "b"}),
        )
        data = store.get("tok")
        self.assertEqual(data["messages"][0], {"role": "user", "content": "q1"})
        self.assertEqual(data["messages"][1]["role"], "assistant")
        self.assertEqual(data["messages"][1]["content"], "a1")
        self.assertEqual(data["messages"][1]["route"], "r")
        self.assertEqual(data["messages"][1]["response"], {"body": "b"})

    # ------------------------------------------------------------------
    # with_state support
    # ------------------------------------------------------------------

    def test_with_state_old_data_defaults_empty(self):
        store = self._store()
        captured = {}

        def gen(h, state):
            captured["state"] = state
            return make_result("a1", state={"count": 1})

        ret = store.turn("tok", "r1", 0, "q1", gen, with_state=True)
        self.assertEqual(captured["state"], {})
        self.assertEqual(ret["result"]["state"], {"count": 1})

    def test_with_state_survives_more_than_three_turns(self):
        store = self._store()
        for i in range(6):
            store.turn(
                "tok", "r%d" % i, i, "q%d" % i,
                lambda h, s, i=i: make_result("a%d" % i, state={"count": i + 1}),
                with_state=True,
            )
        captured = {}

        def gen(h, state):
            captured["state"] = state
            return make_result("final", state={"count": 99})

        store.turn("tok", "r6", 6, "q6", gen, with_state=True)
        self.assertEqual(captured["state"], {"count": 6})

    def test_with_state_restart(self):
        store = self._store()
        store.turn(
            "tok", "r1", 0, "q1",
            lambda h, s: make_result("a1", state={"count": 1}),
            with_state=True,
        )
        store.close()
        store2 = self._store()
        captured = {}

        def gen(h, state):
            captured["state"] = state
            return make_result("a2", state={"count": 2})

        store2.turn("tok", "r2", 1, "q2", gen, with_state=True)
        self.assertEqual(captured["state"], {"count": 1})

    def test_with_state_clear_resets(self):
        store = self._store()
        store.turn(
            "tok", "r1", 0, "q1",
            lambda h, s: make_result("a1", state={"count": 1}),
            with_state=True,
        )
        store.clear("tok")
        captured = {}

        def gen(h, state):
            captured["state"] = state
            return make_result("a2", state={"count": 2})

        store.turn("tok", "r2", 2, "q2", gen, with_state=True)
        self.assertEqual(captured["state"], {})

    def test_with_state_expiry_resets(self):
        store = self._store(ttl_seconds=0.1)
        store.turn(
            "tok", "r1", 0, "q1",
            lambda h, s: make_result("a1", state={"count": 1}),
            with_state=True,
        )
        time.sleep(0.3)
        captured = {}

        def gen(h, state):
            captured["state"] = state
            return make_result("a2", state={"count": 2})

        store.turn("tok", "r2", 0, "q2", gen, with_state=True)
        self.assertEqual(captured["state"], {})

    def test_with_state_isolation(self):
        store = self._store()
        store.turn(
            "alice", "r1", 0, "q1",
            lambda h, s: make_result("a1", state={"who": "alice"}),
            with_state=True,
        )
        store.turn(
            "bob", "r1", 0, "q1",
            lambda h, s: make_result("b1", state={"who": "bob"}),
            with_state=True,
        )
        captured = {}

        def gen(h, state):
            captured["state"] = state
            return make_result("a2", state={"who": "alice2"})

        store.turn("alice", "r2", 1, "q2", gen, with_state=True)
        self.assertEqual(captured["state"], {"who": "alice"})

    def test_with_state_duplicate_request_does_not_update(self):
        store = self._store()
        calls = []

        def gen(h, state):
            calls.append(1)
            return make_result("a1", state={"count": 1})

        first = store.turn("tok", "r1", 0, "q1", gen, with_state=True)
        second = store.turn("tok", "r1", 0, "q1", gen, with_state=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["result"]["state"], {"count": 1})
        self.assertEqual(second["result"]["state"], {"count": 1})
        captured = {}

        def gen2(h, state):
            captured["state"] = state
            return make_result("a2", state={"count": 2})

        store.turn("tok", "r2", 1, "q2", gen2, with_state=True)
        self.assertEqual(captured["state"], {"count": 1})

    def test_with_state_callback_failure_no_write(self):
        store = self._store()

        def bad_gen(h, state):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            store.turn("tok", "r1", 0, "q1", bad_gen, with_state=True)
        data = store.get("tok")
        self.assertEqual(data["revision"], 0)
        self.assertEqual(data["messages"], [])
        ok = store.turn(
            "tok", "r1", 0, "q1",
            lambda h, s: make_result("a1", state={"count": 1}),
            with_state=True,
        )
        self.assertEqual(ok["revision"], 1)

    def test_with_state_illegal_result_no_write(self):
        store = self._store()
        with self.assertRaises(TypeError):
            store.turn("tok", "r1", 0, "q1", lambda h, s: "nope", with_state=True)
        with self.assertRaises(TypeError):
            store.turn(
                "tok", "r1", 0, "q1",
                lambda h, s: make_result("a1"),
                with_state=True,
            )
        with self.assertRaises(TypeError):
            store.turn(
                "tok", "r1", 0, "q1",
                lambda h, s: make_result("a1", state="nope"),
                with_state=True,
            )
        data = store.get("tok")
        self.assertEqual(data["revision"], 0)
        self.assertEqual(data["messages"], [])

    def test_with_state_history_has_no_state(self):
        store = self._store()
        store.turn(
            "tok", "r1", 0, "q1",
            lambda h, s: make_result("a1", route="r1", state={"count": 1}),
            with_state=True,
        )
        captured = {}

        def gen(h, state):
            captured["history"] = h
            return make_result("a2", state={"count": 2})

        store.turn("tok", "r2", 1, "q2", gen, with_state=True)
        hist = captured["history"]
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0], {"role": "user", "content": "q1"})
        self.assertEqual(hist[1], {"role": "assistant", "content": "a1", "route": "r1"})
        self.assertNotIn("state", hist[1])

    def test_with_state_messages_hide_state(self):
        store = self._store()
        store.turn(
            "tok", "r1", 0, "q1",
            lambda h, s: make_result(
                "a1", route="r", response={"body": "b"}, state={"count": 1}
            ),
            with_state=True,
        )
        data = store.get("tok")
        self.assertEqual(data["messages"][1]["role"], "assistant")
        self.assertEqual(data["messages"][1]["content"], "a1")
        self.assertEqual(data["messages"][1]["route"], "r")
        self.assertEqual(data["messages"][1]["response"], {"body": "b"})
        self.assertNotIn("state", data["messages"][1])

    def test_with_state_mutation_does_not_pollute(self):
        store = self._store()
        store.turn(
            "tok", "r1", 0, "q1",
            lambda h, s: make_result("a1", state={"count": 1, "items": [1]}),
            with_state=True,
        )

        def gen(h, state):
            state["count"] = 999
            state["items"].append(999)
            return make_result("a2", state={"count": 2})

        store.turn("tok", "r2", 1, "q2", gen, with_state=True)
        # the state passed to gen was a deep copy: r1's stored state is intact
        store.close()
        conn = sqlite3.connect(self.path)
        try:
            row = conn.execute(
                "SELECT result_json FROM turns WHERE request_id = 'r1'"
            ).fetchone()
            self.assertEqual(
                json.loads(row[0])["state"], {"count": 1, "items": [1]}
            )
        finally:
            conn.close()
        # mutating the returned result copy does not pollute the stored state
        ret = store.turn(
            "tok", "r3", 2, "q3",
            lambda h, s: make_result("a3", state={"count": 3}),
            with_state=True,
        )
        ret["result"]["state"]["count"] = 777
        cached = store.turn(
            "tok", "r3", 2, "q3",
            lambda h, s: make_result("never"),
            with_state=True,
        )
        self.assertEqual(cached["result"]["state"], {"count": 3})

    def test_with_state_false_keeps_old_contract(self):
        store = self._store()
        store.turn(
            "tok", "r1", 0, "q1",
            lambda h, s: make_result("a1", state={"count": 1}),
            with_state=True,
        )
        captured = {}

        def gen(h):
            captured["history"] = h
            return make_result("a2")

        ret = store.turn("tok", "r2", 1, "q2", gen)  # with_state defaults False
        self.assertEqual(len(captured["history"]), 2)
        self.assertEqual(
            captured["history"][1],
            {"role": "assistant", "content": "a1", "route": "default"},
        )
        self.assertNotIn("state", captured["history"][1])
        self.assertNotIn("state", ret["result"])


if __name__ == "__main__":
    unittest.main()