import unittest
from unittest.mock import patch

import app


class CompanionFlowTests(unittest.TestCase):
    def run_query(self, query, docs=None, history=None, generated=None):
        with patch.object(app, "classify_intent", return_value=None), patch.object(app, "call_companion_llm", return_value=generated), patch.object(app, "embedding", return_value=None):
            return app.answer(query, docs or [], history or [])

    def test_vague_input_gets_companion_response(self):
        result = self.run_query("有些困扰")
        self.assertEqual(result["route"], "companion")
        self.assertNotIn("超出知识库", result["answer"])

    def test_general_input_is_not_hard_refused(self):
        result = self.run_query("我能休息休息吗")
        self.assertEqual(result["route"], "companion")
        self.assertTrue(result["answer"].startswith("可以"))

    def test_knowledge_is_added_when_evidence_exists(self):
        docs = [{"id": "d1", "name": "feeding.md", "chunks": [{"id": "c1", "text": "婴儿满6月龄左右开始添加辅食。"}]}]
        result = self.run_query("什么时候开始添加辅食？", docs, generated="婴儿满6月龄左右开始添加辅食。[1]")
        self.assertEqual(result["route"], "companion_with_knowledge")
        self.assertEqual(len(result["citations"]), 1)

    def test_no_model_never_returns_raw_knowledge(self):
        docs = [{"id": "d1", "name": "feeding.md", "chunks": [{"id": "c1", "text": "婴儿满6月龄左右开始添加辅食。"}]}]
        result = self.run_query("什么时候开始添加辅食？", docs)
        self.assertEqual(result["knowledge_status"], "generation_unavailable")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("6月龄", result["answer"])

    def test_medical_question_without_evidence_gets_boundary_not_refusal(self):
        result = self.run_query("宝宝皮疹怎么办")
        self.assertEqual(result["route"], "companion")
        self.assertNotEqual(result["status"], "refused")
        self.assertIn("当前知识库没有足够依据", result["answer"])

    def test_knowledge_without_evidence_does_not_call_generator(self):
        plan = {"emotion_present": False, "emotion": "", "need": "information", "knowledge_present": True, "knowledge_query": "宝宝皮疹", "strategy": "knowledge_support"}
        with patch.object(app, "classify_intent", return_value=plan), patch.object(app, "embedding", return_value=None), patch.object(app, "call_companion_llm", return_value="知识库外医疗建议") as companion:
            result = app.answer("宝宝皮疹怎么办", [])
        companion.assert_not_called()
        self.assertNotIn("知识库外医疗建议", result["answer"])

    def test_product_recommendation_gets_natural_boundary(self):
        result = self.run_query("推荐什么奶粉")
        self.assertEqual(result["route"], "companion")
        self.assertIn("不太适合替你直接选具体品牌", result["answer"])

    def test_emergency_still_preempts_companion(self):
        result = self.run_query("宝宝呼吸困难怎么办")
        self.assertEqual(result["route"], "medical_urgent")

    def test_crisis_still_preempts_companion(self):
        result = self.run_query("我不想活了")
        self.assertEqual(result["route"], "mental_health_crisis")

    def test_identity_is_fixed_capability(self):
        result = self.run_query("你是谁？")
        self.assertEqual(result["route"], "assistant_identity")
        self.assertIn("母婴陪伴助手", result["answer"])

    def test_capabilities_are_fixed(self):
        result = self.run_query("你会些什么？")
        self.assertEqual(result["route"], "assistant_identity")
        self.assertIn("主要能做两件事", result["answer"])

    def test_doctor_identity_is_denied(self):
        result = self.run_query("你是医生吗？")
        self.assertEqual(result["route"], "assistant_identity")
        self.assertTrue(result["answer"].startswith("不是"))

    def test_medication_request_uses_professional_boundary(self):
        result = self.run_query("宝宝能吃药吗？", generated="可以吃布洛芬")
        self.assertEqual(result["route"], "medical_boundary")
        self.assertNotIn("布洛芬", result["answer"])

    def test_medical_boundary_precedes_identity_in_compound_query(self):
        result = self.run_query("你是医生吗，宝宝能吃药吗？")
        self.assertEqual(result["route"], "medical_boundary")

    def test_short_reply_continues_companion_context(self):
        history = [{"role": "assistant", "content": "今天已经够辛苦了。", "route": "companion"}]
        result = self.run_query("好的", history=history)
        self.assertEqual(result["route"], "companion")
        self.assertEqual(result["answer"], "嗯，那就先让自己缓一会儿，不急着继续说。我在这里。")

    def test_first_person_followup_continues_context(self):
        history = [{"role": "assistant", "content": "这种被占满的感觉很消耗。", "route": "companion"}]
        result = self.run_query("我能休息休息吗", history=history, generated="可以，先给自己一点喘息。")
        self.assertEqual(result["route"], "companion")
        self.assertEqual(result["answer"], "可以，先给自己一点喘息。")


if __name__ == "__main__":
    unittest.main()
