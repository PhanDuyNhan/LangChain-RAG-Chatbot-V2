# =============================================================
# src/rag_chain.py - RAG PIPELINE CHÍNH (MỨC 1)
# =============================================================
# Luồng xử lý:
#   Câu hỏi → [Cache check] → Embed query → ChromaDB search (top_k=4)
#           → Ghép context → System Prompt → LLM (+ retry backoff)
#           → Parse JSON → [Lưu cache] → Trả về
#
# Tối ưu quota tích hợp sẵn:
#   - ResponseCache   : câu hỏi lặp lại = 0 token
#   - RateLimiter     : không bao giờ vượt 14 RPM Gemini
#   - Exponential Backoff: tự retry khi 429 trước khi fallback
#
# ĐÃ SỬA:
#   - Cache LLM instance per provider → không init lại mỗi query
#   - Chỉ fallback khi gặp 429/quota, không loop khởi tạo mỗi lần
#   - Dùng cached LLM nếu provider hiện tại còn hoạt động
# =============================================================

import os
import sys
import json
import re
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain.schema import Document, HumanMessage

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_router import get_llm, get_current_provider
from ingest import COLLECTION_NAME, CHROMA_DB_PATH, check_db_exists, load_vector_store, create_embeddings

try:
    from quota_guard import (
        get_rag_cache,
        get_llm_limiter,
        get_counter,
        get_daily_tracker,
        GEMINI_LLM_RPD_SOFT,
    )
    _rag_cache   = get_rag_cache()
    _llm_limiter = get_llm_limiter()
    _counter     = get_counter()
    _daily_tracker = get_daily_tracker()
except ImportError:
    _rag_cache = _llm_limiter = _counter = _daily_tracker = None

load_dotenv()

# =============================================================
# SYSTEM PROMPT
# =============================================================
# Thiết kế với 5 nguyên tắc:
#   1. Vai trò rõ ràng   → tránh lạc đề
#   2. CHỈ dùng context  → chống hallucination
#   3. JSON schema cứng  → ít lỗi format
#   4. Citation bắt buộc → có thể kiểm tra lại
#   5. Confidence level  → người dùng biết độ tin cậy
# =============================================================
SYSTEM_PROMPT = """Bạn là trợ lý hỗ trợ sinh viên của Trường Đại học Sài Gòn (SGU).

NHIỆM VỤ: Trả lời câu hỏi về quy định học vụ, thủ tục, học phí, học bổng của SGU.

NGUYÊN TẮC:
1. CHỈ dùng thông tin trong [CONTEXT] bên dưới.
2. KHÔNG bịa đặt. Nếu không có thông tin, trả lời: "Tôi không tìm thấy thông tin này."
3. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu.
4. Nếu câu hỏi hỏi về điều kiện, thủ tục, hồ sơ, quy trình, học phí hoặc nhiều ý, hãy trả lời ĐẦY ĐỦ các ý chính trong context.
5. Ưu tiên trả lời dạng gạch đầu dòng hoặc đánh số khi có nhiều điều kiện/bước.
6. Độ dài vừa phải, thường tối đa khoảng 1500 ký tự; không lan man ngoài context.

[CONTEXT]
{context}

[CÂU HỎI]
{question}

Hãy trả lời trực tiếp cho sinh viên bằng văn bản thuần.
Không dùng JSON. Không dùng markdown code block.
Nếu có nhiều ý, trình bày rõ ràng theo dòng hoặc đánh số."""


def _strip_markdown_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    return text


def _extract_balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _extract_json_fragment(text: str) -> Optional[dict]:
    cleaned = _strip_markdown_json(text)
    candidates = [cleaned]

    balanced = _extract_balanced_json(cleaned)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        if "{" in candidate:
            try:
                obj, _ = decoder.raw_decode(candidate[candidate.find("{"):])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


def _extract_string_field(text: str, key: str) -> Optional[str]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
    if not match:
        return None
    value = match.group(1)
    value = bytes(value, "utf-8").decode("unicode_escape")
    return value.strip()


def _extract_list_field(text: str, key: str) -> list:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    inner = match.group(1)
    return [
        bytes(m, "utf-8").decode("unicode_escape").strip()
        for m in re.findall(r'"((?:\\.|[^"\\])*)"', inner)
        if m.strip()
    ]


def _normalize_rag_payload(data: dict, depth: int = 0) -> dict:
    """
    Gemini 2.5 Flash đôi khi trả JSON lồng trong field "answer".
    Hàm này bóc thêm một tầng để UI hiển thị câu trả lời sạch hơn.
    """
    if not isinstance(data, dict):
        return {
            "answer": str(data).strip(),
            "source": "Không xác định",
            "confidence": "low",
            "related_topics": [],
        }

    answer = data.get("answer", "Không có câu trả lời.")
    source = data.get("source", "Không xác định")
    confidence = str(data.get("confidence", "low")).lower().strip()
    related_topics = data.get("related_topics", [])

    if depth < 2 and isinstance(answer, str):
        nested = _extract_json_fragment(answer)
        if isinstance(nested, dict) and nested.get("answer"):
            merged = {
                "answer": nested.get("answer", answer),
                "source": nested.get("source", source),
                "confidence": nested.get("confidence", confidence),
                "related_topics": nested.get("related_topics", related_topics),
            }
            return _normalize_rag_payload(merged, depth + 1)

        if answer.startswith('{"answer"') or answer.startswith("{\\\"answer\\\""):
            nested_answer = _extract_string_field(answer, "answer")
            nested_source = _extract_string_field(answer, "source")
            nested_conf = _extract_string_field(answer, "confidence")
            nested_topics = _extract_list_field(answer, "related_topics")
            if nested_answer:
                return _normalize_rag_payload(
                    {
                        "answer": nested_answer,
                        "source": nested_source or source,
                        "confidence": nested_conf or confidence,
                        "related_topics": nested_topics or related_topics,
                    },
                    depth + 1,
                )

    if isinstance(answer, str):
        answer = answer.strip()
        if answer.startswith("{"):
            stripped = re.sub(r'^\{\s*"answer"\s*:\s*"', "", answer).strip()
            stripped = re.sub(r'"\s*,?\s*$', "", stripped).strip()
            stripped = re.sub(r'"\s*\}\s*$', "", stripped).strip()
            if stripped and stripped != answer:
                answer = bytes(stripped, "utf-8").decode("unicode_escape").strip()

        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1].strip()

    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    if not isinstance(related_topics, list):
        related_topics = []

    return {
        "answer": str(answer).strip(),
        "source": str(source).strip() or "Không xác định",
        "confidence": confidence,
        "related_topics": related_topics,
    }


def _clean_answer_text(answer: str) -> str:
    text = (answer or "").strip()
    text = _strip_markdown_json(text)

    if text.startswith("{") and '"answer"' in text:
        nested = _extract_json_fragment(text)
        if isinstance(nested, dict) and nested.get("answer"):
            return _clean_answer_text(str(nested.get("answer", "")))

        text = re.sub(r'^\{\s*"answer"\s*:\s*"', "", text).strip()
        text = re.sub(r'"\s*,?\s*$', "", text).strip()
        text = re.sub(r'"\s*\}\s*$', "", text).strip()

    text = text.replace("\\n", "\n").replace('\\"', '"').strip()
    text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[\{\[]+", "", text).strip()
    return text.strip(' "\'')


def _build_source_from_docs(docs: list) -> str:
    by_file: Dict[str, list] = {}
    for doc in docs[:3]:
        filename = doc.metadata.get("filename", "Tài liệu SGU")
        page = str(doc.metadata.get("page", "?"))
        by_file.setdefault(filename, [])
        if page not in by_file[filename]:
            by_file[filename].append(page)

    labels = []
    for filename, pages in by_file.items():
        if len(pages) == 1:
            labels.append(f"Trang {pages[0]} ({filename})")
        else:
            labels.append(f"Trang {', '.join(pages)} ({filename})")

    return " | ".join(labels) if labels else "Không xác định"


def _estimate_confidence(answer: str, docs: list) -> str:
    if not docs:
        return "low"
    answer_len = len((answer or "").strip())
    top_doc_len = len((docs[0].page_content or "").strip()) if docs else 0
    if answer_len >= 40 and top_doc_len >= 80:
        return "high"
    if answer_len >= 15:
        return "medium"
    return "low"


def _build_related_topics(question: str, docs: list, max_items: int = 4) -> list:
    """
    Tạo related_topics ổn định để response JSON có thể dùng cho UI / lưu DB
    mà không cần thêm một request Gemini nữa.
    """
    topics = []
    q = (question or "").strip()
    q_lower = q.lower()
    question_words = {
        w for w in re.findall(r"\w+", q_lower, flags=re.UNICODE)
        if len(w) >= 3
    }

    if q:
        topics.append(q.rstrip(" ?"))

    for doc in docs[:3]:
        content = (doc.page_content or "").replace("\n", " ")
        matches = re.findall(r"KB\d{3}[A-Z]?\s+(.{5,80}?)\s*(?:Tr\.|Q:|A:|$)", content)
        for match in matches:
            topic = re.sub(r"\s+", " ", match).strip(" -:.")
            if not topic or topic in topics:
                continue

            topic_lower = topic.lower()
            if question_words:
                overlap = sum(1 for w in question_words if w in topic_lower)
                if overlap == 0:
                    continue

            if topic and topic not in topics:
                topics.append(topic)

    deduped = []
    for topic in topics:
        if topic and topic not in deduped:
            deduped.append(topic)

    return deduped[:max_items]


# =============================================================
# CLASS RAGChain
# =============================================================
class RAGChain:

    def __init__(self, force_provider: str = None):
        self.top_k = int(os.getenv("RETRIEVER_TOP_K", "4"))
        # Cache LLM instances → không init lại mỗi lần query
        self._llm_cache: Dict[str, Any] = {}
        self._init_components(force_provider)

    def _init_components(self, force_provider: str = None):
        self.embeddings = create_embeddings()

        if not check_db_exists():
            raise RuntimeError(
                "ChromaDB chưa được tạo!\n"
                "Hãy chạy: python src/ingest.py"
            )

        self.vector_store = load_vector_store(self.embeddings)
        self.retriever    = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        # Khởi tạo LLM đầu tiên và cache lại
        self.llm, self.provider_name = self._get_or_init_llm(force_provider or "gemini")

    def _get_or_init_llm(self, provider_key: str):
        """
        Lấy LLM từ cache hoặc khởi tạo mới.
        Tránh init lại object mỗi lần query → tiết kiệm thời gian.
        """
        if provider_key in self._llm_cache:
            llm, name = self._llm_cache[provider_key]
            return llm, name
        try:
            llm, name = get_llm(force_provider=provider_key, allow_fallback=False)
            self._llm_cache[provider_key] = (llm, name)
            return llm, name
        except Exception as e:
            raise RuntimeError(f"Không khởi tạo được {provider_key}: {e}")

    def _format_context(self, docs: list) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "Tài liệu SGU")
            page     = doc.metadata.get("page", "?")
            parts.append(f"[{i}] Trang {page}, {filename}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _parse_json_response(self, raw: str) -> dict:
        """
        Parse JSON từ LLM với 4 lớp fallback:
          1. JSON thuần túy
          2. Strip markdown rồi parse
          3. Regex tìm JSON trong text
          4. Plain text làm answer
        """
        text = _strip_markdown_json(raw)

        data = _extract_json_fragment(text)
        if isinstance(data, dict):
            return _normalize_rag_payload(data)

        # Lớp 3: bóc field ngay cả khi JSON bị cụt đuôi
        answer = _extract_string_field(text, "answer")
        source = _extract_string_field(text, "source")
        confidence = _extract_string_field(text, "confidence")
        related_topics = _extract_list_field(text, "related_topics")
        if answer:
            return _normalize_rag_payload({
                "answer": answer,
                "source": source or "Không xác định",
                "confidence": confidence or "medium",
                "related_topics": related_topics,
            })

        # Lớp 4: plain text fallback
        lines = raw.strip().split("\n")
        clean = "\n".join(
            l for l in lines
            if not re.match(r"^\[\d+\]\s+Trang\s+\d+", l.strip())
        ).strip() or raw

        return {"answer": clean, "source": "Không xác định", "confidence": "low", "related_topics": []}

    def query(self, question: str) -> Dict[str, Any]:
        """
        Luồng query đầy đủ với tối ưu quota:
          1. Cache check   → trả ngay nếu có
          2. Vector search → lấy top_k chunks
          3. LLM call      → dùng cached LLM, rate limit + retry backoff
          4. Fallback      → chỉ khi bị 429/quota
          5. Parse JSON    → lưu cache
        """
        # ── Bước 1: Cache check ──
        if _rag_cache:
            cached = _rag_cache.get(question)
            if cached:
                if _counter:
                    _counter.increment("cache_hits")
                result = dict(cached)
                result["from_cache"] = True
                return result

        # ── Bước 2: Vector Search ──
        retrieved_docs = self.retriever.invoke(question)
        if not retrieved_docs:
            return {
                "answer":         "Không tìm thấy thông tin liên quan trong tài liệu. "
                                  "Vui lòng liên hệ trực tiếp các phòng ban của trường SGU.",
                "source":         "Không có",
                "confidence":     "low",
                "related_topics": [],
                "provider":       self.provider_name,
                "retrieved_docs": [],
                "from_cache":     False,
            }

        # ── Bước 3: LLM Call với cached instance ──
        context     = self._format_context(retrieved_docs)
        full_prompt = SYSTEM_PROMPT.format(context=context, question=question)
        raw_answer  = None

        # Thứ tự fallback: Gemini → Groq → Ollama
        # Dùng cached LLM nếu có, chỉ init mới khi cần fallback
        fallback_order = [
            ("gemini", 3),   # 3 retry với backoff
            ("groq",   1),   # 1 retry
            ("ollama", 0),   # 0 retry
        ]

        for provider_key, max_retries in fallback_order:
            try:
                # Dùng cached LLM hoặc init mới nếu chưa có
                llm, pname = self._get_or_init_llm(provider_key)
            except Exception as e:
                print(f"[RAG] Không khởi tạo được {provider_key}: {str(e)[:60]}")
                continue

            success = False
            for attempt in range(max_retries + 1):
                try:
                    if provider_key == "gemini" and _daily_tracker:
                        if not _daily_tracker.can_consume("gemini_llm", GEMINI_LLM_RPD_SOFT):
                            print("[RAG] Gemini đạt soft cap theo ngày → fallback tiếp theo")
                            break
                    if provider_key == "gemini" and _llm_limiter:
                        with _llm_limiter:
                            response = llm.invoke([HumanMessage(content=full_prompt)])
                    else:
                        response = llm.invoke([HumanMessage(content=full_prompt)])

                    raw_answer         = response.content
                    self.provider_name = pname
                    if _counter:
                        _counter.increment("gemini_llm" if provider_key == "gemini" else "groq")
                    if provider_key == "gemini" and _daily_tracker:
                        _daily_tracker.increment("gemini_llm")
                    success = True
                    break

                except Exception as e:
                    err    = str(e)
                    is_429 = any(k in err for k in [
                        "429", "quota", "ResourceExhausted", "RESOURCE_EXHAUSTED",
                    ])
                    if is_429 and attempt < max_retries:
                        delay = 2.0 * (2 ** attempt)  # 2s → 4s → 8s
                        print(f"[RAG] {pname} 429, retry sau {delay:.0f}s "
                              f"({attempt+1}/{max_retries})...")
                        time.sleep(delay)
                    elif is_429:
                        # Xóa khỏi cache để lần sau init lại (có thể quota đã reset)
                        self._llm_cache.pop(provider_key, None)
                        print(f"[RAG] {pname} hết quota → fallback tiếp theo")
                        break
                    else:
                        raise  # Lỗi không phải 429 → raise ngay

            if success:
                break

        if raw_answer is None:
            return {
                "answer":         "Tất cả LLM provider đều hết quota hoặc lỗi. "
                                  "Vui lòng thử lại sau hoặc kiểm tra API key trong .env",
                "source":         "Không có",
                "confidence":     "low",
                "related_topics": [],
                "provider":       self.provider_name,
                "retrieved_docs": [],
                "from_cache":     False,
            }

        # ── Bước 4: Parse/Clean + Cache ──
        parsed = self._parse_json_response(raw_answer)
        answer_text = _clean_answer_text(parsed.get("answer", raw_answer))
        if not answer_text:
            answer_text = _clean_answer_text(raw_answer)

        parsed = {
            "answer": answer_text or "Tôi không tìm thấy thông tin phù hợp trong tài liệu.",
            "source": _build_source_from_docs(retrieved_docs),
            "confidence": _estimate_confidence(answer_text, retrieved_docs),
            "related_topics": parsed.get("related_topics") or _build_related_topics(question, retrieved_docs),
            "provider": self.provider_name,
            "from_cache": False,
            "output_format": "json_schema_normalized",
            "schema_version": "rag_v1",
        }
        parsed["retrieved_docs"] = [
            {
                "content":  doc.page_content[:200] + "...",
                "page":     doc.metadata.get("page", "?"),
                "filename": doc.metadata.get("filename", "?"),
            }
            for doc in retrieved_docs
        ]

        if _rag_cache:
            # Cache bản không có retrieved_docs để tiết kiệm memory
            cache_val = {k: v for k, v in parsed.items() if k != "retrieved_docs"}
            _rag_cache.set(cache_val, question)

        return parsed

    def switch_provider(self, provider: str):
        self.llm, self.provider_name = self._get_or_init_llm(provider)


# =============================================================
# SINGLETON
# =============================================================
_instance: Optional[RAGChain] = None


def get_rag_chain(force_provider: str = None) -> RAGChain:
    global _instance
    if _instance is None:
        _instance = RAGChain(force_provider)
    elif force_provider:
        _instance.switch_provider(force_provider)
    return _instance


# =============================================================
# TEST: python src/rag_chain.py
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST RAG CHAIN")
    print("=" * 60)
    chain = RAGChain()
    for q in [
        "Thủ tục bảo lưu kết quả học tập gồm mấy bước?",
        "Học phí ngành Công nghệ thông tin năm 2025 là bao nhiêu?",
        "Điều kiện để được học bổng khuyến khích học tập?",
    ]:
        print(f"\n❓ {q}")
        r = chain.query(q)
        print(f"💬 {r['answer'][:200]}...")
        print(f"📄 {r['source']}  |  📊 {r['confidence']}  |  🤖 {r['provider']}")
        print(f"⚡ Cache: {r.get('from_cache', False)}")
