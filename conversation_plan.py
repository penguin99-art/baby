"""Validated model hints. These fields never grant tool or safety permissions."""

from dataclasses import asdict, dataclass
import json
import re


@dataclass(frozen=True)
class ConversationPlan:
    emotion_present: bool
    knowledge_present: bool
    emotion: str = ""
    knowledge_query: str = ""
    need: str = "other"
    strategy: str = "reflect"

    @classmethod
    def parse_text(cls, text):
        if not isinstance(text, str):
            raise ValueError("plan response must be text")
        raw = text.strip()
        fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1)
        return cls.parse(json.loads(raw))

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("invalid plan fields")
        if any(type(value.get(key)) is not bool for key in ("emotion_present", "knowledge_present")):
            raise ValueError("plan flags must be booleans")
        for key, limit in (("emotion", 30), ("knowledge_query", 300)):
            text = value.get(key, "")
            if not isinstance(text, str) or len(text) > limit:
                raise ValueError("invalid plan text")
        if value.get("need", "other") not in (
            "validation", "listening", "practical_help", "information", "reassurance", "other"
        ):
            raise ValueError("invalid need")
        if value.get("strategy", "reflect") not in (
            "stay_with_feeling", "reflect", "gentle_question", "small_step", "knowledge_support"
        ):
            raise ValueError("invalid strategy")
        plan = cls(**value)
        if not plan.knowledge_present and plan.knowledge_query:
            raise ValueError("knowledge query without knowledge intent")
        return asdict(plan)
