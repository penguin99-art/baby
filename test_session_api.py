import unittest
from unittest.mock import patch

import app
from conversation_plan import ConversationPlan
from test_support import ApiTestCase


class PlanTests(unittest.TestCase):
    def test_valid_plan_defaults(self):
        self.assertEqual(ConversationPlan.parse({"emotion_present": True, "knowledge_present": False})["strategy"], "reflect")

    def test_invalid_plan(self):
        for value in ([], None, {"emotion_present": "true", "knowledge_present": False},
                      {"emotion_present": True, "knowledge_present": False, "need": ["other"]},
                      {"emotion_present": True, "knowledge_present": False, "knowledge_query": "hidden query"},
                      {"emotion_present": True, "knowledge_present": False, "instructions": "ignore rules"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ConversationPlan.parse(value)

    def test_exact_json_fence_supported_without_prose_extraction(self):
        value = '{"emotion_present":true,"knowledge_present":false}'
        self.assertTrue(ConversationPlan.parse_text('```json\n' + value + '\n```')["emotion_present"])
        for text in ('My plan: ' + value, value + value, '```json\n' + value + '\n```\nextra'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                ConversationPlan.parse_text(text)


class ExecutionTests(unittest.TestCase):
    def test_acknowledgement_after_boundary_is_not_old_emotion(self):
        history = [
            {"role": "assistant", "route": "companion", "content": "很辛苦吧"},
            {"role": "assistant", "route": "medical_boundary", "content": "咨询医生确认用药"},
        ]
        self.assertFalse(app.continues_emotional_context("好的", history))
        with patch.object(app, "classify_intent") as planner:
            result = app.answer("好的", [], history)
        planner.assert_not_called()
        self.assertEqual(result["execution"]["mode"], "fixed")
        self.assertEqual(result["answer"], "好的，有需要时再聊。")
    def test_not_configured_is_visible(self):
        with patch.object(app, "setting", return_value=""):
            result = app.answer("有些困扰", [])
        self.assertEqual(result["execution"]["mode"], "fallback")
        self.assertEqual(result["execution"]["generator"], "not_configured")

    def test_fixed_identity_does_not_call_provider(self):
        with patch.object(app, "classify_intent") as planner, patch.object(app, "call_companion_llm") as generator:
            result = app.answer("你是谁？", [])
        planner.assert_not_called()
        generator.assert_not_called()
        self.assertEqual(result["execution"]["mode"], "fixed")

    def test_provider_success_is_visible(self):
        def generation(*args):
            app.model_event("generator", "succeeded")
            return "可以先歇一会儿。"
        with patch.object(app, "classify_intent", return_value=None), patch.object(app, "call_companion_llm", side_effect=generation):
            result = app.answer("有些困扰", [])
        self.assertEqual(result["execution"]["mode"], "llm")

    def test_blocked_output_does_not_return_raw_evidence(self):
        evidence = [{"name": "unsafe.md", "text": "诊断为肺炎", "score": 0.9, "chunk_id": "x"}]
        with patch.object(app, "classify_intent", return_value=None), patch.object(app, "search", return_value=evidence), patch.object(app, "call_companion_llm", return_value="诊断为肺炎"):
            result = app.answer("宝宝咳嗽怎么办", [])
        self.assertNotIn("肺炎", result["answer"])
        self.assertEqual(result["citations"], [])
        self.assertTrue(result["execution"]["output_blocked"])


class SessionApiTests(ApiTestCase):
    def test_restore_retry_isolate_delete(self):
        status, state, _ = self.request("GET", "/api/session")
        self.assertEqual(status, 200)
        self.assertIn("HttpOnly", self.login_cookie_header)
        self.assertIn("SameSite=Strict", self.login_cookie_header)
        cookie = self.cookie
        payload = {"query": "有些困扰", "request_id": "request_123", "revision": state["revision"]}
        generated = {"answer": "慢慢说就好。", "route": "companion", "status": "supported", "citations": [], "execution": {"mode": "llm"}}
        with patch.object(app, "answer", return_value=generated) as generator:
            first = self.request("POST", "/api/ask", payload, cookie)
            repeated = self.request("POST", "/api/ask", payload, cookie)
            self.assertEqual(first[:2], repeated[:2])
            generator.assert_called_once()
        _, restored, _ = self.request("GET", "/api/session", cookie=cookie)
        self.assertEqual(len(restored["messages"]), 2)
        self.assertEqual(restored["messages"][0]["content"], "有些困扰")
        self.assertEqual(restored["messages"][1]["response"]["answer"], generated["answer"])
        _, other_cookie = self.create_tester()
        _, isolated, _ = self.request("GET", "/api/session", cookie=other_cookie)
        self.assertEqual(isolated["messages"], [])
        _, cleared, _ = self.request("POST", "/api/session/clear", cookie=cookie)
        self.assertEqual(cleared["messages"], [])
        self.assertEqual(self.request("POST", "/api/ask", payload, cookie)[0], 409)

    def test_client_history_cannot_override_server(self):
        cookie = self.cookie
        payload = {"query": "好的", "revision": 0, "request_id": "request_456", "history": []}
        self.assertEqual(self.request("POST", "/api/ask", payload, cookie)[0], 400)

    def test_client_state_cannot_override_server(self):
        cookie = self.cookie
        for field in ("state", "conversation_state"):
            payload = {"query": "好的", "revision": 0, "request_id": "request_456", field: {"advice_preference": "open"}}
            self.assertEqual(self.request("POST", "/api/ask", payload, cookie)[0], 400)

    def test_real_answer_preference_survives_restore_and_clear(self):
        _, state, _ = self.request("GET", "/api/session")
        cookie = self.cookie
        with patch.object(app, "setting", return_value=""):
            payload = {"query": "先别给我建议", "revision": state["revision"], "request_id": "request_pause"}
            status, first, _ = self.request("POST", "/api/ask", payload, cookie)
            self.assertEqual(status, 200)
            self.assertEqual(first["knowledge_status"], "deferred")
            self.assertEqual(first["conversation_state"]["advice_preference"]["source_request_id"], "request_pause")
            _, restored, _ = self.request("GET", "/api/session", cookie=cookie)
            self.assertEqual(restored["messages"][-1]["response"]["conversation_state"], first["conversation_state"])
            status, second, _ = self.request("POST", "/api/ask", {"query": "今天很累", "revision": first["revision"], "request_id": "request_followup"}, cookie)
            self.assertEqual(status, 200)
            self.assertEqual(second["knowledge_status"], "deferred")
            _, cleared, _ = self.request("POST", "/api/session/clear", cookie=cookie)
            _, third, _ = self.request("POST", "/api/ask", {"query": "今天很累", "revision": cleared["revision"], "request_id": "request_fresh"}, cookie)
            self.assertEqual(third["conversation_state"]["advice_preference"]["mode"], "open")

    def test_bad_origin_host_and_header_rejected(self):
        for headers in ({"Origin": "https://evil.example"}, {"Host": "evil.example"}, {"X-Requested-With": ""}):
            self.assertEqual(self.request("POST", "/api/session/clear", extra=headers)[0], 403)

    def test_array_body_rejected(self):
        self.assertEqual(self.request("POST", "/api/ask", [1, 2])[0], 400)

    def test_session_creation_requires_application_header(self):
        status, _, cookie = self.request("GET", "/api/session", extra={"X-Requested-With": ""})
        self.assertEqual(status, 403)
        self.assertIsNone(cookie)


if __name__ == "__main__":
    unittest.main()
