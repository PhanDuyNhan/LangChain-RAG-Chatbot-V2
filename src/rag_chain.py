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
import unicodedata
from typing import Dict, Any, Optional, Tuple, List
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

GENERAL_FALLBACK_PROMPT = """Bạn là trợ lý AI hỗ trợ người dùng bằng tiếng Việt.

Câu hỏi dưới đây KHÔNG khớp rõ với tài liệu SGU hiện có, vì vậy hãy trả lời bằng kiến thức chung.

NGUYÊN TẮC:
1. Trả lời trung thực, rõ ràng, dễ hiểu.
2. Nếu câu hỏi cần thông tin chính thức của SGU mà bạn không chắc, hãy nhắc người dùng kiểm tra tài liệu hoặc website chính thức của trường.
3. Có thể dùng gạch đầu dòng hoặc đánh số nếu có nhiều ý.
4. Không nhắc đến JSON. Không dùng code block.

[CÂU HỎI]
{question}

Hãy trả lời trực tiếp bằng văn bản thuần."""

_VI_STOPWORDS = {
    "ban", "toi", "tui", "cho", "cac", "nhung", "nhu", "la", "gi", "nao",
    "sao", "voi", "ve", "cua", "mot", "nhieu", "duoc", "khong", "trong",
    "ngoai", "tai", "lieu", "sgu", "em", "anh", "chi", "va", "theo",
    "kh", "biet", "biết", "ro", "rõ", "ko", "hong",
}

_GENERIC_DOC_TERMS = {
    "sgu", "truong", "dai", "hoc", "sinh", "vien", "phong", "khoa",
    "co", "so", "truonghoc", "daihoc", "sinhvien",
}


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
        self.min_relevance = float(os.getenv("RAG_MIN_RELEVANCE", "0.32"))
        self.cache_namespace = os.getenv("RAG_CACHE_NAMESPACE", "rag_auto_fallback_v3")
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
        #self.llm, self.provider_name = self._get_or_init_llm(force_provider or "gemini")
        self.llm, self.provider_name = self._get_or_init_llm(force_provider or "groq")
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

    def _normalize_distance_score(self, score: Any) -> float:
        try:
            value = max(0.0, float(score))
        except Exception:
            return 0.0
        return max(0.0, min(1.0, 1.0 / (1.0 + value)))

    def _normalize_match_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", (text or "").lower())
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        return normalized

    def _extract_content_terms(self, question: str) -> List[str]:
        normalized_question = self._normalize_match_text(question)
        terms = []
        for word in re.findall(r"\w+", normalized_question, flags=re.UNICODE):
            if len(word) < 3 or word in _VI_STOPWORDS:
                continue
            if word not in terms:
                terms.append(word)
        return terms

    def _analyze_document_match(self, question: str, docs: list) -> Dict[str, Any]:
        terms = self._extract_content_terms(question)
        if not terms or not docs:
            return {
                "terms": terms,
                "specific_terms": [],
                "generic_matches": [],
                "specific_matches": [],
                "specific_overlap": 0.0,
            }

        doc_text = self._normalize_match_text(
            " ".join((doc.page_content or "") for doc in docs[:self.top_k])
        )
        generic_matches = []
        specific_matches = []
        specific_terms = [term for term in terms if term not in _GENERIC_DOC_TERMS]

        for term in terms:
            if term in doc_text:
                if term in _GENERIC_DOC_TERMS:
                    generic_matches.append(term)
                else:
                    specific_matches.append(term)

        specific_overlap = 0.0
        if specific_terms:
            specific_overlap = len(specific_matches) / max(1, len(specific_terms))

        return {
            "terms": terms,
            "specific_terms": specific_terms,
            "generic_matches": generic_matches,
            "specific_matches": specific_matches,
            "specific_overlap": max(0.0, min(1.0, specific_overlap)),
        }

    def _keyword_overlap_score(self, question: str, docs: list) -> float:
        analysis = self._analyze_document_match(question, docs)
        terms = analysis["terms"]
        if not terms or not docs:
            return 0.0

        doc_text = self._normalize_match_text(
            " ".join((doc.page_content or "") for doc in docs[:self.top_k])
        )
        overlap = sum(1 for term in terms if term in doc_text)
        ratio = overlap / max(1, len(terms))
        boosted_ratio = max(ratio, analysis["specific_overlap"])
        return max(0.0, min(1.0, boosted_ratio))

    def _should_general_fallback(
        self,
        question: str,
        docs: list,
        best_relevance: float,
        score_source: str,
    ) -> Tuple[bool, str]:
        if not docs:
            return True, "no_documents"

        analysis = self._analyze_document_match(question, docs)
        specific_terms = analysis["specific_terms"]
        specific_matches = analysis["specific_matches"]
        specific_overlap = analysis["specific_overlap"]

        if not specific_terms and best_relevance < 0.48:
            return True, "generic_match_only"

        if len(specific_terms) >= 2 and not specific_matches:
            return True, "missing_specific_terms"

        if len(specific_terms) >= 2 and specific_overlap < 0.5 and best_relevance < 0.55:
            return True, "weak_specific_match"

        if best_relevance < self.min_relevance:
            return True, "low_relevance"

        if score_source == "keyword_overlap" and specific_terms and specific_overlap < 0.4:
            return True, "keyword_only_match"

        return False, "rag"

    def _retrieve_docs_with_relevance(self, question: str) -> Tuple[list, float, list, str]:
        """
        Trả về:
        - retrieved_docs
        - best_relevance: 0..1, càng cao càng liên quan
        - relevance_scores: list score đã chuẩn hóa
        - score_source: nguồn score dùng để quyết định fallback
        """
        try:
            doc_scores = self.vector_store.similarity_search_with_relevance_scores(
                question, k=self.top_k
            )
            if doc_scores:
                docs = [doc for doc, _ in doc_scores]
                scores = [
                    max(0.0, min(1.0, float(score)))
                    for _, score in doc_scores
                    if score is not None
                ]
                keyword_score = self._keyword_overlap_score(question, docs)
                best = max(scores) if scores else 0.0
                best = max(best, keyword_score)
                return docs, best, scores, "relevance_scores"
        except Exception:
            pass

        try:
            doc_scores = self.vector_store.similarity_search_with_score(
                question, k=self.top_k
            )
            if doc_scores:
                docs = [doc for doc, _ in doc_scores]
                scores = [
                    self._normalize_distance_score(score)
                    for _, score in doc_scores
                    if score is not None
                ]
                keyword_score = self._keyword_overlap_score(question, docs)
                best = max(scores) if scores else 0.0
                best = max(best, keyword_score)
                return docs, best, scores, "distance_scores"
        except Exception:
            pass

        docs = self.retriever.invoke(question)
        keyword_score = self._keyword_overlap_score(question, docs)
        return docs, keyword_score, [], "keyword_overlap"

    def _invoke_with_fallback(self, prompt: str, fallback_order: List[Tuple[str, int]]) -> Tuple[Optional[str], str]:
        raw_answer = None
        provider_name = self.provider_name

        for provider_key, max_retries in fallback_order:
            try:
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
                            response = llm.invoke([HumanMessage(content=prompt)])
                    else:
                        response = llm.invoke([HumanMessage(content=prompt)])

                    raw_answer = response.content
                    provider_name = pname
                    self.provider_name = pname
                    if _counter and provider_key in {"gemini", "groq"}:
                        _counter.increment("gemini_llm" if provider_key == "gemini" else "groq")
                    if provider_key == "gemini" and _daily_tracker:
                        _daily_tracker.increment("gemini_llm")
                    success = True
                    break

                except Exception as e:
                    err = str(e)
                    is_429 = any(k in err for k in [
                        "429", "quota", "ResourceExhausted", "RESOURCE_EXHAUSTED",
                    ])
                    if is_429 and attempt < max_retries:
                        delay = 2.0 * (2 ** attempt)
                        print(f"[RAG] {pname} 429, retry sau {delay:.0f}s ({attempt+1}/{max_retries})...")
                        time.sleep(delay)
                    elif is_429:
                        self._llm_cache.pop(provider_key, None)
                        print(f"[RAG] {pname} hết quota → fallback tiếp theo")
                        break
                    else:
                        raise

            if success:
                break

        return raw_answer, provider_name

    def _general_fallback_response(
        self,
        question: str,
        reason: str,
        relevance_score: float,
        related_topics: Optional[list] = None,
    ) -> Dict[str, Any]:
        prompt = GENERAL_FALLBACK_PROMPT.format(question=question)
        raw_answer, provider_name = self._invoke_with_fallback(
            prompt,
            [
                ("groq", 1),
                ("gemini", 3),
                ("ollama", 0),
            ],
        )

        if raw_answer is None:
            return {
                "answer": "Tôi chưa thể trả lời câu hỏi ngoài tài liệu ở thời điểm này. Vui lòng thử lại sau.",
                "source": "Ngoài tài liệu SGU",
                "confidence": "low",
                "related_topics": related_topics or [question.strip().rstrip(" ?")],
                "provider": self.provider_name,
                "retrieved_docs": [],
                "from_cache": False,
                "response_mode": "general_fallback",
                "fallback_reason": reason,
                "relevance_score": round(relevance_score, 3),
                "output_format": "json_schema_normalized",
                "schema_version": "rag_v1",
            }

        answer_text = _clean_answer_text(raw_answer)
        confidence = "medium" if len(answer_text) >= 40 else "low"

        return {
            "answer": answer_text or "Tôi chưa thể trả lời câu hỏi ngoài tài liệu ở thời điểm này.",
            "source": "Ngoài tài liệu SGU (kiến thức chung)",
            "confidence": confidence,
            "related_topics": related_topics or [question.strip().rstrip(" ?")],
            "provider": provider_name,
            "retrieved_docs": [],
            "from_cache": False,
            "response_mode": "general_fallback",
            "fallback_reason": reason,
            "relevance_score": round(relevance_score, 3),
            "output_format": "json_schema_normalized",
            "schema_version": "rag_v1",
        }

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
        Luồng query Mức 1:
          1. Cache check
          2. Retrieve + đo độ liên quan
          3. Nếu liên quan thấp → auto fallback sang chat ngoài tài liệu
          4. Nếu liên quan đủ cao → chạy RAG như bình thường
          5. Chuẩn hóa output + lưu cache
        """
        question = (question or "").strip()
        if not question:
            return {
                "answer": "Vui lòng nhập câu hỏi trước khi tìm kiếm.",
                "source": "Không có",
                "confidence": "low",
                "related_topics": [],
                "provider": self.provider_name,
                "retrieved_docs": [],
                "from_cache": False,
                "response_mode": "rag",
                "relevance_score": 0.0,
                "relevance_scores": [],
                "score_source": "none",
                "output_format": "json_schema_normalized",
                "schema_version": "rag_v1",
            }

        if _rag_cache:
            cached = _rag_cache.get(self.cache_namespace, question)
            if cached:
                if _counter:
                    _counter.increment("cache_hits")
                result = dict(cached)
                result["from_cache"] = True
                return result

        retrieved_docs, best_relevance, relevance_scores, score_source = self._retrieve_docs_with_relevance(question)
        related_topics = _build_related_topics(question, retrieved_docs) or [question.rstrip(" ?")]
        retrieved_preview = [
            {
                "content": doc.page_content[:200] + "...",
                "page": doc.metadata.get("page", "?"),
                "filename": doc.metadata.get("filename", "?"),
            }
            for doc in retrieved_docs
        ]
        rounded_scores = [round(float(score), 3) for score in relevance_scores[:self.top_k]]
        rounded_best_relevance = round(float(best_relevance), 3)

        should_fallback, fallback_reason = self._should_general_fallback(
            question, retrieved_docs, best_relevance, score_source
        )
        if should_fallback:
            response = self._general_fallback_response(
                question=question,
                reason=fallback_reason,
                relevance_score=best_relevance,
                related_topics=related_topics,
            )
            response["retrieved_docs"] = retrieved_preview if retrieved_docs else []
            response["relevance_scores"] = rounded_scores if retrieved_docs else []
            response["score_source"] = score_source
            response["matched_doc_count"] = len(retrieved_docs or [])
            if _rag_cache:
                _rag_cache.set(
                    {k: v for k, v in response.items() if k != "retrieved_docs"},
                    self.cache_namespace,
                    question,
                )
            return response

        context = self._format_context(retrieved_docs)
        full_prompt = SYSTEM_PROMPT.format(context=context, question=question)
        raw_answer, provider_name = self._invoke_with_fallback(
            full_prompt,
            [
                ("groq", 1),
                ("gemini", 3),
                ("ollama", 0),
            ],
        )

        if raw_answer is None:
            return {
                "answer": "Tất cả LLM provider đều hết quota hoặc đang lỗi. Vui lòng thử lại sau hoặc kiểm tra API key trong .env.",
                "source": _build_source_from_docs(retrieved_docs) if retrieved_docs else "Không có",
                "confidence": "low",
                "related_topics": related_topics,
                "provider": self.provider_name,
                "retrieved_docs": retrieved_preview,
                "from_cache": False,
                "response_mode": "rag",
                "relevance_score": rounded_best_relevance,
                "relevance_scores": rounded_scores,
                "score_source": score_source,
                "matched_doc_count": len(retrieved_docs),
                "output_format": "json_schema_normalized",
                "schema_version": "rag_v1",
            }

        parsed = self._parse_json_response(raw_answer)
        answer_text = _clean_answer_text(parsed.get("answer", raw_answer))
        if not answer_text:
            answer_text = _clean_answer_text(raw_answer)

        response = {
            "answer": answer_text or "Tôi không tìm thấy thông tin phù hợp trong tài liệu.",
            "source": _build_source_from_docs(retrieved_docs),
            "confidence": _estimate_confidence(answer_text, retrieved_docs),
            "related_topics": parsed.get("related_topics") or related_topics,
            "provider": provider_name,
            "retrieved_docs": retrieved_preview,
            "from_cache": False,
            "response_mode": "rag",
            "relevance_score": rounded_best_relevance,
            "relevance_scores": rounded_scores,
            "score_source": score_source,
            "matched_doc_count": len(retrieved_docs),
            "output_format": "json_schema_normalized",
            "schema_version": "rag_v1",
        }

        if _rag_cache:
            _rag_cache.set(
                {k: v for k, v in response.items() if k != "retrieved_docs"},
                self.cache_namespace,
                question,
            )

        return response

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
