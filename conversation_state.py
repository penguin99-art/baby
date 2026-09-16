"""Explicit, session-scoped preferences; never medical facts or tool permissions."""

import re


_QUOTED = re.compile(r'“[^”]*”|「[^」]*」|"[^"\n]*"|\'[^\'\n]*\'')
_DEFER = re.compile(
    r"(?:请)?(?:先|暂时|现在)?(?:不要|别|不用)(?:再)?(?:给我|给|提|讲|说|提供|解释)"
    r"(?:一些|任何|这些|那些)?(?:建议|方法|办法|解决方案)(?:了|吧)?"
    r"|(?:我)?(?:现在|暂时)?不想(?:听|要|看)(?:你的|任何)?(?:建议|方法|办法)(?:了)?"
    r"|(?:先|暂时)不(?:讲|说)(?:方法|建议|办法)(?:了|吧)?"
    r"|(?:我)?(?:现在|暂时)?只想(?:倾诉|说说|聊聊|让你听我说)"
)
_RESUME = re.compile(
    r"(?:那|请)?(?:现在|这次)?(?:可以)?(?:给我|讲讲|说说|提供)(?:一些|点|一点)?"
    r"(?:建议|方法|办法)(?:了|吧|吗)?"
    r"|(?:我)?(?:现在)?想(?:听听|听|了解)(?:你的|一些)?(?:建议|方法|办法)(?:了)?"
    r"|(?:请)?(?:现在)?帮我(?:想想|想个)(?:办法|方法)"
)
_CLOSE = re.compile(r"(?:谢谢)?(?:我想)?(?:先)?(?:聊到这里|到这里|不聊了|结束聊天)(?:吧|了)?")


def _clauses(query):
    text = _QUOTED.sub("", query)
    return [part.strip() for part in re.split(r"[，,。.!！？?；;\n]", text) if part.strip()]


def preference_change(query):
    """Only complete, explicit clauses change the preference; last clause wins."""
    change = None
    for clause in _clauses(query):
        if _DEFER.fullmatch(clause):
            change = "deferred"
        elif _RESUME.fullmatch(clause):
            change = "open"
    return change


def update_state(previous, query, request_id=None):
    if not isinstance(previous, dict):
        raise ValueError("invalid conversation state")
    preference = {"mode": "open", "source_request_id": None}
    if previous:
        if set(previous) != {"version", "advice_preference"} or type(previous["version"]) is not int or previous["version"] != 1:
            raise ValueError("invalid state version")
        stored = previous["advice_preference"]
        if not isinstance(stored, dict) or set(stored) != set(preference):
            raise ValueError("invalid preference")
        source = stored["source_request_id"]
        if stored["mode"] not in ("open", "deferred") or (source is not None and (not isinstance(source, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", source))):
            raise ValueError("invalid preference value")
        preference = dict(stored)
    change = preference_change(query)
    if change is not None:
        preference = {"mode": change, "source_request_id": request_id}
    return {"version": 1, "advice_preference": preference}


def wants_to_close(query):
    clauses = _clauses(query)
    return bool(clauses) and all(clause in {"谢谢", "谢谢你", "好的", "好"} or _CLOSE.fullmatch(clause) for clause in clauses) and any(_CLOSE.fullmatch(clause) for clause in clauses)


def asks_for_information(query):
    text = "，".join(_clauses(query))
    return bool(re.search(r"怎么|如何|什么时候|多少|是否|能不能|可不可以|有什么|哪些|为什么|请问|想了解", text) or "?" in query or "？" in query)


def listening_reply(query, history=None):
    if preference_change(query) == "deferred":
        return "好，先不讲方法，也不给建议。我会听你说。"
    if re.fullmatch(r"[好嗯谢你的是，。！？!?\s]+", query):
        return "嗯，我在听。"
    if any(word in query for word in ("累", "疲惫", "辛苦")):
        return "你说的这种累，我听见了。这里不急着找办法。"
    if any(word in query for word in ("内疚", "自责", "委屈")):
        return "这些感受可以慢慢说，不需要急着把它们解释清楚。"
    text = "我在听，你可以按自己的节奏说。"
    latest = next((item.get("content") for item in reversed(history or []) if item.get("role") == "assistant"), None)
    return "不用急着找到答案，我会认真听你说。" if latest == text else text
