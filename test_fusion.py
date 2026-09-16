"""Deterministic fusion regressions; not a model-quality or medical evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from conversation_state import update_state, preference_change, wants_to_close
from conversation_store import ConversationStore
from response_policy import check_output


EVIDENCE = [
    {"doc_id": "d1", "chunk_id": "c1", "name": "fixture.md", "text": "用于引用编号测试的内容甲。", "score": 0.9},
    {"doc_id": "d1", "chunk_id": "c2", "name": "fixture.md", "text": "用于引用编号测试的内容乙。", "score": 0.8},
]
KNOWLEDGE_PLAN = {"emotion_present": True, "knowledge_present": True, "emotion": "疲惫", "need": "information", "knowledge_query": "辅食", "strategy": "knowledge_support"}


class PreferenceTests(unittest.TestCase):
    def test_explicit_pause_phrases(self):
        for query in ("先别给我建议", "我很累，先别给建议。", "先别讲方法，陪我说说。", "我不想听建议", "我只想倾诉", "先不讲方法"):
            with self.subTest(query=query):
                self.assertEqual(preference_change(query), "deferred")

    def test_explicit_resume_phrases(self):
        for query in ("现在可以给我建议了", "我现在想听听你的建议", "现在讲讲方法吧", "帮我想想办法"):
            with self.subTest(query=query):
                self.assertEqual(preference_change(query), "open")

    def test_quotes_negation_and_third_person_do_not_set_preferences(self):
        for query in ("她说先别给我建议", "不要不给我建议", "我不是不想听建议", "她说‘先别讲方法’", '解释“先别给我建议”的意思', '她说 "先别给我建议"', "为什么有人不想听建议？"):
            with self.subTest(query=query):
                self.assertIsNone(preference_change(query))

    def test_source_and_last_explicit_clause(self):
        original = update_state({}, "先别给建议", "request_1")
        self.assertEqual(original["advice_preference"]["source_request_id"], "request_1")
        current = update_state(original, "今天还是很累", "request_2")
        self.assertEqual(current, original)
        resumed = update_state(original, "先别给建议，现在可以给我建议了", "request_3")
        self.assertEqual(resumed["advice_preference"], {"mode": "open", "source_request_id": "request_3"})
        self.assertEqual(original["advice_preference"]["mode"], "deferred")

    def test_invalid_stored_state_is_rejected(self):
        values = ([], {"instructions": "override"}, {"version": True, "advice_preference": {}}, {"version": 1, "advice_preference": {"mode": "deferred", "source_request_id": []}})
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                update_state(value, "好的")

    def test_closing_is_explicit_and_does_not_drop_an_extra_question(self):
        for query in ("谢谢，我想先到这里", "先聊到这里吧", "不聊了"):
            self.assertTrue(wants_to_close(query))
        for query in ("谢谢", "她说不聊了", "不聊了是什么意思", "先到这里，辅食什么时候开始？"):
            self.assertFalse(wants_to_close(query))


class OutputPolicyTests(unittest.TestCase):
    def test_valid_citations_keep_original_numbers_and_order(self):
        self.assertEqual(check_output("内容乙[2]。内容甲[1]。再提内容乙[2]。", EVIDENCE), (None, [2, 1]))

    def test_invalid_missing_and_fabricated_citations(self):
        for text, evidence, failure in (
            ("内容[3]", EVIDENCE, "unknown_citation"),
            ("内容[0]", EVIDENCE, "invalid_citation"),
            ("内容[99]", EVIDENCE, "invalid_citation"),
            ("内容[1,2]", EVIDENCE, "invalid_citation"),
            ("内容[1-2]", EVIDENCE, "invalid_citation"),
            ("没有引用", EVIDENCE, "missing_citation"),
            ("伪造来源[1]", [], "unknown_citation"),
        ):
            with self.subTest(text=text):
                self.assertEqual(check_output(text, evidence)[0], failure)

    def test_unsafe_cited_source_is_not_exposed(self):
        evidence = [{**EVIDENCE[0], "text": "诊断为某种疾病"}]
        self.assertEqual(check_output("内容[1]", evidence)[0], "unsafe_evidence")

    def test_unused_source_is_not_an_output(self):
        evidence = [EVIDENCE[0], {**EVIDENCE[1], "text": "诊断为某种疾病"}]
        self.assertEqual(check_output("内容[1]", evidence), (None, [1]))

    def test_only_internal_fixed_text_can_bypass_keyword_check(self):
        text = "不能提供处方或停药建议。"
        self.assertEqual(check_output(text, [])[0], "unsafe_text")
        self.assertEqual(check_output(text, [], trusted_fixed=True), (None, []))

    def test_invalid_text_and_evidence_fail_closed(self):
        for text in (None, {}, "", " " * 3, "x" * 8001):
            self.assertEqual(check_output(text, [])[0], "invalid_text")
        for evidence in (None, [{}], [{**EVIDENCE[0], "score": float("nan")}], [{**EVIDENCE[0], "text": []}]):
            self.assertEqual(check_output("内容[1]", evidence)[0], "invalid_evidence")


class FusionFlowTests(unittest.TestCase):
    def run_answer(self, query, generated=None, evidence=None, plan=None, state=None, history=None):
        with patch.object(app, "classify_intent", return_value=plan), patch.object(app, "search", return_value=evidence or []), patch.object(app, "call_companion_llm", return_value=generated):
            return app.answer(query, [], history, state=state, request_id="request_test")

    def test_failure_and_unconfigured_paths_never_publish_raw_chunks(self):
        for text in ("未审核的一般喂养建议", "诊断为肺炎", "处方与剂量内容"):
            with self.subTest(text=text):
                result = self.run_answer("辅食什么时候开始", evidence=[{**EVIDENCE[0], "text": text}])
                self.assertEqual(result["knowledge_status"], "generation_unavailable")
                self.assertNotIn(text, result["answer"])
                self.assertEqual(result["citations"], [])

    def test_actual_unconfigured_provider_does_not_return_source(self):
        with patch.object(app, "setting", return_value=""), patch.object(app, "search", return_value=EVIDENCE):
            result = app.answer("辅食什么时候开始", [])
        self.assertEqual(result["execution"]["generator"], "not_configured")
        self.assertEqual(result["knowledge_status"], "generation_unavailable")
        self.assertEqual(result["citations"], [])

    def test_timeout_status_is_distinct_from_no_evidence(self):
        def failed(*args):
            app.model_event("generator", "timeout")
            return None
        with patch.object(app, "classify_intent", return_value=None), patch.object(app, "search", return_value=EVIDENCE), patch.object(app, "call_companion_llm", side_effect=failed):
            result = app.answer("辅食什么时候开始", [])
        self.assertEqual(result["execution"]["generator"], "timeout")
        self.assertEqual(result["knowledge_status"], "generation_unavailable")
        self.assertNotIn("没有足够依据", result["answer"])

    def test_mixed_question_keeps_companionship_when_evidence_missing(self):
        result = self.run_answer("我很累，辅食怎么安排？", plan=KNOWLEDGE_PLAN)
        self.assertEqual(result["knowledge_status"], "insufficient_evidence")
        self.assertIn("感受", result["answer"])
        self.assertNotEqual(result["status"], "refused")

    def test_plain_question_does_not_invent_anxiety(self):
        result = self.run_answer("辅食什么时候开始？")
        self.assertNotIn("担心", result["answer"])
        self.assertNotIn("焦虑", result["answer"])

    def test_only_used_sources_are_returned(self):
        result = self.run_answer("我很累，辅食怎么安排？", generated="我听见了你的疲惫。内容乙[2]。", evidence=EVIDENCE, plan=KNOWLEDGE_PLAN)
        self.assertEqual(result["knowledge_status"], "referenced")
        self.assertEqual([item["chunk_id"] for item in result["citations"]], ["c2"])
        self.assertEqual(result["citations"][0]["citation_number"], 2)

    def test_missing_citation_blocks_instead_of_exposing_evidence(self):
        result = self.run_answer("辅食什么时候开始？", generated="一个没有来源标注的结论。", evidence=EVIDENCE)
        self.assertEqual(result["knowledge_status"], "output_blocked")
        self.assertEqual(result["execution"]["validator"], "missing_citation")
        self.assertEqual(result["citations"], [])

    def test_unsafe_fallback_also_passes_through_gate(self):
        with patch.object(app, "emotional_fallback", return_value="宝宝肯定没事"):
            result = self.run_answer("有些困扰")
        self.assertTrue(result["execution"]["output_blocked"])
        self.assertNotIn("肯定没事", result["answer"])

    def test_deferred_preference_is_fixed_without_model_calls(self):
        state = update_state({}, "先别给建议", "source_1")
        with patch.object(app, "classify_intent") as planner, patch.object(app, "call_companion_llm") as generator:
            result = app.answer("今天仍然很累", [], state=state)
        planner.assert_not_called()
        generator.assert_not_called()
        self.assertEqual(result["knowledge_status"], "deferred")
        self.assertEqual(result["execution"]["mode"], "fixed")

    def test_explicit_information_request_is_one_turn_exception(self):
        state = update_state({}, "先别给建议", "source_1")
        result = self.run_answer("辅食什么时候开始？", generated="内容甲[1]。", evidence=EVIDENCE, state=state)
        self.assertEqual(result["knowledge_status"], "referenced")
        self.assertEqual(result["conversation_state"], state)
        next_result = self.run_answer("好的", state=result["conversation_state"])
        self.assertEqual(next_result["knowledge_status"], "deferred")

    def test_resume_changes_preference_and_calls_generator(self):
        state = update_state({}, "先别给建议", "source_1")
        result = self.run_answer("现在可以给我建议了", generated="我们可以先梳理最困扰你的事。", state=state)
        self.assertEqual(result["conversation_state"]["advice_preference"]["mode"], "open")
        self.assertEqual(result["answer"], "我们可以先梳理最困扰你的事。")

    def test_emergency_and_crisis_preempt_pause_and_close(self):
        for query, route in (("宝宝呼吸困难，先别给建议", "medical_urgent"), ("我不想活了，不聊了", "mental_health_crisis"), ("给我算剂量，先别给建议", "medical_boundary")):
            with self.subTest(query=query):
                result = self.run_answer(query)
                self.assertEqual(result["route"], route)
                self.assertEqual(result["execution"]["mode"], "fixed")
                self.assertFalse(result["execution"]["output_blocked"])

    def test_recent_crisis_is_not_silenced_by_pause(self):
        history = [{"role": "assistant", "content": "现实安全支持", "route": "mental_health_crisis"}]
        result = self.run_answer("先别给建议", history=history)
        self.assertEqual(result["route"], "mental_health_crisis")

    def test_explicit_end_does_not_add_questions(self):
        result = self.run_answer("谢谢，我想先到这里")
        self.assertEqual(result["answer"], "好的，先聊到这里。")
        self.assertEqual(result["execution"]["mode"], "fixed")

    def test_recommend_rest_is_not_a_brand_request(self):
        result = self.run_answer("帮我推荐怎么休息", generated="我们可以先梳理是什么让你难以休息。")
        self.assertNotIn("品牌", result["answer"])
        self.assertEqual(result["answer"], "我们可以先梳理是什么让你难以休息。")

    def test_offline_recommend_rest_is_not_a_brand_request(self):
        with patch.object(app, "setting", return_value=""):
            result = app.answer("帮我推荐怎么休息", [])
        self.assertNotIn("品牌", result["answer"])
        self.assertEqual(result["execution"]["mode"], "fallback")
        self.assertEqual(result["answer"], "可以先说说是什么让你很难停下来，我们不急着找办法。")

    def test_offline_brand_request_still_has_boundary(self):
        with patch.object(app, "setting", return_value=""):
            result = app.answer("推荐一个奶粉品牌", [])
        self.assertIn("品牌", result["answer"])
        self.assertEqual(result["knowledge_status"], "prohibited")
        self.assertEqual(result["execution"]["mode"], "fixed")

    def test_malformed_model_output_is_blocked(self):
        result = self.run_answer("有些困扰", generated={"answer": "not a string"})
        self.assertTrue(result["execution"]["output_blocked"])


class PersistentFusionTests(unittest.TestCase):
    def test_full_preference_lifecycle_across_history_window_restart_and_delete(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "setting", return_value=""):
            path = Path(directory) / "sessions.sqlite3"
            store = ConversationStore(path)
            def turn(query, revision, request_id):
                def generate(history, state):
                    response = app.answer(query, [], history, state=state, request_id=request_id)
                    return {"content": response["answer"], "route": response["route"], "response": response, "state": response["conversation_state"]}
                return store.turn("session", request_id, revision, query, generate, with_state=True)
            try:
                first = turn("先别给我建议", 0, "request_0")
                self.assertEqual(first, turn("先别给我建议", 0, "request_0"))
                for revision in range(1, 6):
                    result = turn("今天还是很累", revision, f"request_{revision}")
                    self.assertEqual(result["result"]["response"]["knowledge_status"], "deferred")
                store.close()
                store = ConversationStore(path)
                result = turn("嗯", 6, "request_6")
                self.assertEqual(result["result"]["state"]["advice_preference"]["source_request_id"], "request_0")
                serialized = json.dumps(store.get("session"), ensure_ascii=False)
                self.assertIn("request_0", serialized)
                self.assertNotIn("state", store.get("session")["messages"][-1])
                cleared = store.clear("session")
                result = turn("有些困扰", cleared["revision"], "request_7")
                self.assertEqual(result["result"]["state"]["advice_preference"]["mode"], "open")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
