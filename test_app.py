import unittest
from unittest.mock import patch

import app


class RoutingTests(unittest.TestCase):
    def run_query(self, query, route=None):
        with patch.object(app, "classify_route", return_value=route), patch.object(app, "call_emotional_llm", return_value=None), patch.object(app, "call_llm", return_value=None), patch.object(app, "embedding", return_value=None):
            return app.answer(query, [])

    def test_vague_emotion_uses_support(self):
        result = self.run_query("有些困扰")
        self.assertEqual(result["route"], "emotional_support")

    def test_caregiver_tired_is_not_medical(self):
        result = self.run_query("照顾宝宝让我有点累")
        self.assertEqual(result["route"], "emotional_support")

    def test_medical_context_cannot_be_swallowed_by_emotion_route(self):
        result = self.run_query("宝宝发烧了，我很焦虑", "emotional_support")
        self.assertEqual(result["route"], "mixed")

    def test_oral_symptom_cannot_be_swallowed_by_emotion_route(self):
        result = self.run_query("宝宝吐了，我好焦虑", "emotional_support")
        self.assertEqual(result["route"], "mixed")

    def test_medication_request_is_refused_before_router(self):
        result = self.run_query("我睡不着了想吃安眠药", "emotional_support")
        self.assertEqual(result["route"], "hard_refusal")

    def test_crisis_is_refused_before_router(self):
        result = self.run_query("我不想活了", "emotional_support")
        self.assertEqual(result["route"], "mental_health_crisis")

    def test_knowledge_evidence_wins_over_out_of_scope_label(self):
        docs = [{"id": "d1", "name": "feeding.md", "chunks": [{"id": "c1", "text": "婴儿满6月龄左右开始添加辅食。"}]}]
        with patch.object(app, "classify_route", return_value="out_of_scope"):
            result = app.answer("什么时候开始添加辅食？", docs)
        self.assertEqual(result["route"], "knowledge")


if __name__ == "__main__":
    unittest.main()
