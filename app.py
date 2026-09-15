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
    base_url = os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "")).rstrip("/")
    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", ""))
    model = os.getenv("EMBEDDING_MODEL", "")
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


def call_llm(query: str, evidence: list[dict]) -> str | None:
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
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


def answer(query: str, docs: list[dict]) -> dict:
    emergency = emergency_message(query)
    if emergency:
        return {"status": "emergency", "answer": emergency, "citations": [], "reason": "命中安全规则"}
    evidence = [item for item in search(query, docs) if item["score"] >= THRESHOLD]
    if not evidence or evidence[0]["score"] < THRESHOLD:
        return {"status": "refused", "answer": "这个问题不在当前母婴知识库的覆盖范围内，或知识库中没有足够证据。为避免误导，我暂不回答。请尝试询问已导入文档中的内容。", "citations": evidence[:2], "reason": f"最高证据分数 {evidence[0]['score'] if evidence else 0:.2f} < 门槛 {THRESHOLD:.2f}"}
    generated = call_llm(query, evidence)
    unsafe = re.compile(r"(mg\s*/?\s*kg|毫克|剂量|每公斤|诊断为|确诊|处方|停药|换药|服用.{0,12}(布洛芬|对乙酰氨基酚|抗生素))", re.I)
    if generated and unsafe.search(generated):
        generated = None
    text = generated or "\n\n".join(f"{item['text']} [{i + 1}]" for i, item in enumerate(evidence[:3]))
    return {"status": "answered", "answer": text, "citations": evidence[:3], "reason": "已通过闭域证据门槛"}


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
