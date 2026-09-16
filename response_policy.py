"""Mechanical output checks, not a semantic or clinical correctness verifier."""

import math
import re


UNSAFE = re.compile(
    r"你有抑郁|你是焦虑症|我能治好|保证会好|只能依赖我|mg\s*/?\s*kg|毫克|每公斤|"
    r"诊断为|确诊|处方|停药|换药|服用.{0,12}(?:布洛芬|对乙酰氨基酚|抗生素)|"
    r"(?:宝宝|孩子|你)(?:肯定|一定|绝对)(?:没事|没有病)|这不是病",
    re.I,
)
_CITATION = re.compile(r"\[(\d+(?:\s*[,，~-]\s*\d+)*)\]")


def check_output(text, citations, *, trusted_fixed=False):
    """Return (failure_code, used_indices). Indices are one-based, never renumbered."""
    if not isinstance(text, str) or not text.strip() or len(text) > 8000:
        return "invalid_text", []
    if not isinstance(citations, list) or len(citations) > 3:
        return "invalid_evidence", []
    if not trusted_fixed and UNSAFE.search(text):
        return "unsafe_text", []
    for item in citations:
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key] for key in ("chunk_id", "name", "text")):
            return "invalid_evidence", []
        score = item.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            return "invalid_evidence", []
    used = []
    for match in _CITATION.finditer(text):
        marker = match.group(1)
        if not re.fullmatch(r"[1-3]", marker):
            return "invalid_citation", []
        index = int(marker)
        if index > len(citations):
            return "unknown_citation", []
        if index not in used:
            used.append(index)
    if citations and not used:
        return "missing_citation", []
    for index in used:
        if UNSAFE.search(citations[index - 1]["text"]):
            return "unsafe_evidence", []
    return None, used
