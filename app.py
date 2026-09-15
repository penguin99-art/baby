from __future__ import annotations

import json
import os
import re
import uuid
import threading
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"
STORE = DATA_DIR / "knowledge.json"
SUPPORTED = {".md"}
# A deliberately conservative gate for the demo: low lexical overlap is not
# enough to authorize an answer in a closed-domain health assistant.
THRESHOLD = 0.24
MAX_UPLOAD = 5 * 1024 * 1024
STORE_LOCK = threading.Lock()
CONFIG_LOCK = threading.Lock()
APP_CONFIG: dict[str, str] = {}


def ensure_store() -> list[dict]:
    DATA_DIR.mkdir(exist_ok=True)
    if not STORE.exists():
        STORE.write_text("[]", encoding="utf-8")
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_store(docs: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    temporary = STORE.with_suffix(".tmp")
    temporary.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STORE)


def terms(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9]+", lowered))
    compact = re.sub(r"\s+", "", re.sub(r"[^\u4e00-\u9fff]", "", lowered))
    chars = set(compact)
    bigrams = {compact[i : i + 2] for i in range(len(compact) - 1)}
    return words | chars | bigrams


def chunks_for(name: str, text: str) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|(?<=[。！？])\s*", text) if p.strip()]
    chunks = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) > 520 and buffer:
            chunks.append(buffer)
            buffer = ""
        buffer += ("\n" if buffer else "") + paragraph
    if buffer:
        chunks.append(buffer)
    return [{"id": f"{name}-{i + 1}", "text": chunk} for i, chunk in enumerate(chunks)]


def embedding(text: str) -> list[float] | None:
    base_url = setting("EMBEDDING_BASE_URL", setting("LLM_BASE_URL")).rstrip("/")
    api_key = setting("EMBEDDING_API_KEY", setting("LLM_API_KEY"))
    model = setting("EMBEDDING_MODEL")
    if not (base_url and api_key and model):
        return None
    body = json.dumps({"model": model, "input": text}).encode()
    request = Request(f"{base_url}/v1/embeddings", data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode())
        return payload["data"][0]["embedding"]
    except Exception:
        return None


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0


def setting(name: str, default: str = "") -> str:
    return APP_CONFIG.get(name, os.getenv(name, default))


def search(query: str, docs: list[dict], limit: int = 4) -> list[dict]:
    query_terms = terms(query)
    query_vector = embedding(query)
    results = []
    for doc in docs:
        title_terms = terms(doc["name"])
        for chunk in doc["chunks"]:
            chunk_terms = terms(chunk["text"])
            overlap = len(query_terms & chunk_terms)
            exact_phrase = any(len(token) > 1 and token in chunk["text"] for token in query_terms)
            vector_score = cosine(query_vector, chunk["embedding"]) if query_vector and chunk.get("embedding") else 0
            if not overlap and vector_score < THRESHOLD:
                continue
            lexical_score = overlap / max(1, len(query_terms))
            if overlap < 2 and not exact_phrase and vector_score < THRESHOLD:
                continue
            score = max(lexical_score, vector_score)
            score += 0.08 if query_terms & title_terms else 0
            results.append({
                "doc_id": doc["id"],
                "name": doc["name"],
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "score": min(score, 0.99),
            })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


def emergency_message(query: str) -> str | None:
    signals = ["呼吸困难", "呼吸有点困难", "喘不上气", "喘不过气", "抽搐", "惊厥", "不吃奶", "拒奶", "不肯吃奶", "意识不清", "叫不醒", "发绀", "嘴唇发紫"]
    if any(signal in query for signal in signals):
        return "这个描述可能涉及需要尽快评估的情况。请立即联系儿科医生或前往最近的急诊；如果出现呼吸困难、意识异常、抽搐或发绀，请拨打 120。系统不能在线判断病因，也不能提供用药剂量。"
    return None


def crisis_message(query: str) -> str | None:
    signals = ["不想活", "想死", "自杀", "伤害自己", "伤害孩子", "杀了孩子", "孩子也别想活", "控制不住自己", "没有人身安全", "家暴"]
    if any(signal in query for signal in signals):
        return "我很重视你刚才说的内容。请现在先不要独处，把自己和孩子交给身边可信任的成年人照看，并立即联系当地急救电话、前往最近的急诊或联系当地心理危机服务。如果已经有人受伤或处于危险中，请马上拨打 120 或当地紧急求助电话。"
    return None


def high_risk_message(query: str) -> str | None:
    signals = ["剂量", "吃什么药", "用什么药", "想吃药", "安眠药", "服什么药", "停药", "换药", "处方", "诊断", "是不是肺炎", "是不是抑郁", "疫苗禁忌", "补种几针"]
    if any(signal in query for signal in signals):
        return "这个问题涉及个体化医疗或心理判断，我不能在线提供诊断、处方、用药剂量、停药建议或疫苗禁忌判断。请咨询儿科医生、妇产科医生、心理专业人员或接种门诊。"
    return None


def is_emotional_support(query: str) -> bool:
    signals = ["有点累", "好累", "很累", "太累了", "累坏了", "疲惫", "压力", "焦虑", "委屈", "难过", "孤单", "孤独", "没人理解", "想哭", "崩溃", "内疚", "自责", "睡不着", "陪我聊", "听我说", "心情", "情绪", "困扰", "烦心"]
    return any(signal in query for signal in signals)


def has_medical_context(query: str) -> bool:
    clinical_signals = ["发烧", "发热", "有点烫", "咳嗽", "呕吐", "吐了", "吐奶", "腹泻", "拉肚子", "不会坐", "发育", "症状", "疼", "出血", "奶量", "不吃奶", "不肯吃奶", "不肯吃饭", "哭闹", "脸色发黄", "黄疸", "湿疹", "拒奶", "呼吸", "疫苗", "辅食"]
    age_or_child = ["宝宝", "婴儿", "孩子", "新生儿", "月龄"]
    return any(signal in query for signal in clinical_signals) or (
        any(signal in query for signal in age_or_child) and any(mark in query for mark in ["怎么办", "怎么", "是否", "能不能", "不会", "异常", "担心"])
    )


def classify_intent(query: str) -> dict | None:
    """Use the configured model for intent hints, never as a safety gate."""
    base_url = setting("LLM_BASE_URL").rstrip("/")
    api_key = setting("LLM_API_KEY")
    model = setting("LLM_MODEL")
    if not (base_url and api_key and model):
        return None
    prompt = (
        "分析用户消息，只输出 JSON，不要解释。字段 emotion_present 和 knowledge_present 必须是 true/false。"
        "emotion_present=true 表示用户在表达感受或寻求倾听；knowledge_present=true 表示在询问母婴知识或照护事实。"
        "购物、品牌推荐、成人话题或其他无关内容的 knowledge_present=false。包含宝宝、婴儿、孩子或月龄，并涉及发烧、发热、咳嗽、呕吐、腹泻、疼痛、出血、拒奶、黄疸、湿疹、呼吸、发育、疫苗或辅食时，knowledge_present必须为true。请不要做安全判断。"
        "格式必须是 {\"emotion_present\":false,\"knowledge_present\":true}。\n用户消息：" + query
    )
    body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
    request = Request(f"{base_url}/v1/chat/completions", data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode())
        raw = payload["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        if not isinstance(parsed.get("emotion_present"), bool) or not isinstance(parsed.get("knowledge_present"), bool):
            return None
        return parsed
    except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError, OSError):
        return None


def emotional_fallback(query: str) -> str:
    if any(word in query for word in ["有点累", "好累", "很累", "太累了", "累坏了", "疲惫", "睡不着"]):
        return "听起来你这段时间真的很累。照顾孩子本来就需要持续投入，有疲惫感并不代表你做得不好，也不代表你不爱孩子。现在不必一次解决所有事情，可以先做一个很小的选择：先喝点水、请可信任的人接手一会儿，或者把最压着你的那件事告诉我。你更想让我先听你说，还是一起梳理下一步？"
    return "听起来你现在承受了不少情绪和压力。你的感受值得被认真听见，不需要急着证明自己足够坚强。我们可以慢一点：你更想让我先陪你倾诉，还是一起把眼前最困扰你的事情拆开？如果你或孩子当下不安全，请立即联系身边可信任的人和当地急救资源。"


def empathy_lead(query: str) -> str:
    if any(word in query for word in ["有点累", "好累", "很累", "太累了", "累坏了", "疲惫"]):
        return "听起来你这段时间真的很累，有这种感受并不代表你做得不好。"
    if any(word in query for word in ["焦虑", "担心", "害怕", "不安"]):
        return "我能理解这件事让你有些担心，照护孩子时出现这样的不安很常见。"
    return "我能理解这件事让你有些困扰，你的感受值得被认真听见。"


def call_llm(query: str, evidence: list[dict]) -> str | None:
    base_url = setting("LLM_BASE_URL").rstrip("/")
    api_key = setting("LLM_API_KEY")
    model = setting("LLM_MODEL")
    if not (base_url and api_key and model):
        return None
    evidence = evidence[:3]
    context = "\n\n".join(f"[{i + 1}] {item['name']}\n{item['text']}" for i, item in enumerate(evidence))
    prompt = (
        "你是封闭域母婴健康科普助手。只能依据下面的知识库证据回答，禁止使用外部知识、推测、诊断、处方或药物剂量。"
        "如果证据不足，只回复‘知识库中没有足够依据回答这个问题’。回答末尾必须保留引用编号。\n\n"
        f"知识库证据：\n{context}\n\n用户问题：{query}"
    )
    body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
    request = Request(f"{base_url}/v1/chat/completions", data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode())
        return payload["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def call_emotional_llm(query: str) -> str | None:
    base_url = setting("LLM_BASE_URL").rstrip("/")
    api_key = setting("LLM_API_KEY")
    model = setting("LLM_MODEL")
    if not (base_url and api_key and model):
        return None
    prompt = (
        "你是面向母亲和照护者的温和情绪支持伙伴。请用简体中文回应，先共情和复述感受，再给一个很小的可选下一步，最后询问用户想倾诉还是梳理。"
        "不要诊断心理疾病，不承诺治愈，不提供药物或医疗建议，不强迫积极，不责备，不暗示用户只能依赖你。"
        "如果用户提到自伤、他伤、儿童伤害、家暴或当下不安全，只回复建议立即联系身边可信任的人、当地急救/急诊和危机服务的安全提示。"
        f"\n用户表达：{query}"
    )
    body = json.dumps({"model": model, "temperature": 0.4, "messages": [{"role": "user", "content": prompt}]}).encode()
    request = Request(f"{base_url}/v1/chat/completions", data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode())
        return payload["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def answer(query: str, docs: list[dict]) -> dict:
    emergency = emergency_message(query)
    if emergency:
        return {"status": "emergency", "route": "medical_urgent", "answer": emergency, "citations": [], "reason": "命中医疗急症规则"}
    crisis = crisis_message(query)
    if crisis:
        return {"status": "crisis", "route": "mental_health_crisis", "answer": crisis, "citations": [], "reason": "命中心理危机规则"}
    high_risk = high_risk_message(query)
    if high_risk:
        return {"status": "refused", "route": "hard_refusal", "answer": high_risk, "citations": [], "reason": "命中高风险主题规则"}
    intent = classify_intent(query)
    medical_context = has_medical_context(query)
    emotion_present = intent["emotion_present"] if intent else is_emotional_support(query)
    knowledge_present = intent["knowledge_present"] if intent else medical_context
    if medical_context:
        knowledge_present = True
    if emotion_present and not knowledge_present:
        generated = call_emotional_llm(query)
        unsafe = re.compile(r"(你有抑郁|你是焦虑症|我能治好|保证会好|只要积极|只能依赖我|处方|剂量|停药|换药|诊断为)", re.I)
        if not generated or unsafe.search(generated):
            generated = emotional_fallback(query)
        return {"status": "supported", "route": "emotional_support", "answer": generated, "citations": [], "reason": "情绪支持通道"}
    evidence = [item for item in search(query, docs) if item["score"] >= THRESHOLD]
    if not evidence or evidence[0]["score"] < THRESHOLD:
        if emotion_present:
            return {"status": "supported", "route": "mixed", "answer": empathy_lead(query) + "\n\n关于其中的母婴知识问题，当前知识库没有足够资料支持回答，建议咨询儿科或儿童保健专业人员。", "citations": [], "reason": "已识别情绪表达，但知识证据不足"}
        return {"status": "refused", "route": "out_of_scope", "answer": "这个问题不在当前母婴知识库的覆盖范围内，且不属于情绪支持。为避免误导，我暂不回答。你可以询问已导入文档中的内容，或和我聊聊最近的感受。", "citations": evidence[:2], "reason": f"最高证据分数 {evidence[0]['score'] if evidence else 0:.2f} < 门槛 {THRESHOLD:.2f}"}
    generated = call_llm(query, evidence)
    unsafe = re.compile(r"(mg\s*/?\s*kg|毫克|剂量|每公斤|诊断为|确诊|处方|停药|换药|服用.{0,12}(布洛芬|对乙酰氨基酚|抗生素))", re.I)
    if generated and unsafe.search(generated):
        generated = None
    text = generated or "\n\n".join(f"{item['text']} [{i + 1}]" for i, item in enumerate(evidence[:3]))
    if emotion_present:
        text = empathy_lead(query) + "\n\n下面补充知识库中有依据的部分：\n\n" + text
    return {"status": "answered", "route": "mixed" if emotion_present else "knowledge", "answer": text, "citations": evidence[:3], "reason": "情绪优先承接，已通过闭域证据门槛" if emotion_present else "已通过闭域证据门槛"}


def public_config() -> dict:
    def masked(value: str) -> str:
        return f"{value[:4]}••••{value[-3:]}" if len(value) > 8 else ("已配置" if value else "")
    return {
        "llm_base_url": setting("LLM_BASE_URL"),
        "llm_model": setting("LLM_MODEL"),
        "llm_api_key": masked(setting("LLM_API_KEY")),
        "embedding_base_url": setting("EMBEDDING_BASE_URL", setting("LLM_BASE_URL")),
        "embedding_model": setting("EMBEDDING_MODEL"),
        "embedding_api_key": masked(setting("EMBEDDING_API_KEY", setting("LLM_API_KEY"))),
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, code: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/docs":
            docs = ensure_store()
            self.send_json({"docs": [{"id": d["id"], "name": d["name"], "chunks": len(d["chunks"])} for d in docs]})
            return
        if path == "/api/config":
            self.send_json(public_config())
            return
        target = (STATIC_DIR / ("index.html" if path == "/" else path.removeprefix("/"))).resolve()
        if target.is_file() and target.is_relative_to(STATIC_DIR.resolve()):
            content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "text/css; charset=utf-8" if target.suffix == ".css" else "application/javascript; charset=utf-8"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.send_json({"error": "invalid content length"}, 400)
            return
        if length > MAX_UPLOAD:
            self.send_json({"error": "upload too large"}, 413)
            return
        body = self.rfile.read(length)
        path = urlparse(self.path).path
        docs = ensure_store()
        if path == "/api/ask":
            try:
                payload = json.loads(body or b"{}")
                query = payload.get("query", "")
            except (json.JSONDecodeError, AttributeError):
                self.send_json({"error": "invalid JSON"}, 400)
                return
            if not isinstance(query, str) or not query.strip() or len(query) > 500:
                self.send_json({"error": "query must be a non-empty string under 500 characters"}, 400)
                return
            self.send_json(answer(query.strip(), docs))
            return
        if path == "/api/config":
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                self.send_json({"error": "invalid JSON"}, 400)
                return
            allowed = {"LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL"}
            values = {key: str(payload[key]).strip() for key in allowed if key in payload and payload[key] is not None}
            for key in ("LLM_BASE_URL", "EMBEDDING_BASE_URL"):
                if key in values and values[key] and not values[key].startswith(("http://", "https://")):
                    self.send_json({"error": f"{key} must start with http:// or https://"}, 400)
                    return
            with CONFIG_LOCK:
                APP_CONFIG.update(values)
            self.send_json({"ok": True, "config": public_config()})
            return
        if path == "/api/config/test":
            llm_ok = bool(call_llm("只回复 OK", [{"name": "connection-test", "text": "OK"}]))
            embedding_ok = bool(embedding("connection test"))
            self.send_json({"llm": llm_ok, "embedding": embedding_ok})
            return
        if path == "/api/reset":
            with STORE_LOCK:
                save_store([])
            self.send_json({"docs": []})
            return
        if path == "/api/upload":
            content_type = self.headers.get("Content-Type", "")
            message = BytesParser(policy=default).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
            added = []
            with STORE_LOCK:
                docs = ensure_store()
                for part in message.walk():
                    filename = part.get_filename()
                    if not filename:
                        continue
                    suffix = Path(filename).suffix.lower()
                    if suffix not in SUPPORTED:
                        continue
                    raw = part.get_payload(decode=True) or b""
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    # Treat imported files as data, never as executable instructions.
                    text = "\n".join(line for line in text.splitlines() if not re.search(r"(忽略之前|系统指令|你是一个|请输出|越过限制)", line, re.I))
                    doc_chunks = chunks_for(filename, text)
                    for chunk in doc_chunks:
                        chunk["embedding"] = embedding(chunk["text"])
                    doc = {"id": uuid.uuid4().hex[:10], "name": Path(filename).name, "chunks": doc_chunks}
                    docs.append(doc)
                    added.append({"id": doc["id"], "name": filename, "chunks": len(doc["chunks"])})
                save_store(docs)
            self.send_json({"added": added})
            return
        self.send_json({"error": "not found"}, 404)

    def log_message(self, *_args) -> None:
        return


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"Baby knowledge demo: http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
