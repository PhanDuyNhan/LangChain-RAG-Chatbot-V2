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
    from quota_guard import get_rag_cache, get_llm_limiter, get_counter
    _rag_cache   = get_rag_cache()
    _llm_limiter = get_llm_limiter()
    _counter     = get_counter()
except ImportError:
    _rag_cache = _llm_limiter = _counter = None

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

⚠️ FORMAT OUTPUT - QUAN TRỌNG:
Chỉ trả về một object JSON duy nhất. KHÔNG viết gì khác ngoài JSON.
KHÔNG dùng markdown. KHÔNG dùng ```json. Chỉ trả về JSON thuần túy.

Schema bắt buộc:
{{"answer":"nội dung trả lời","source":"Trang X, tên file","confidence":"high","related_topics":["topic1"]}}

Ví dụ câu trả lời hợp lệ:
{{"answer":"Điều kiện tốt nghiệp gồm: (1) Không bị truy cứu hình sự. (2) Tích lũy đủ học phần. (3) GPA tích lũy từ 2.0 trở lên. (4) Có chứng chỉ GDQP và hoàn thành GDTC.","source":"Trang 18, cam_nang_sinh_vien.pdf","confidence":"high","related_topics":["điều kiện tốt nghiệp","GPA","học vụ"]}}

confidence: "high" = thông tin rõ ràng | "medium" = không hoàn toàn khớp | "low" = suy luận gián tiếp

[CONTEXT]
{context}

[CÂU HỎI]
{question}

Trả lời (chỉ JSON, không gì khác):"""


# =============================================================
# CLASS RAGChain
# =============================================================
class RAGChain:

    def __init__(self, force_provider: str = None):
        self.top_k = int(os.getenv("RETRIEVER_TOP_K", "4"))
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
        self.llm, self.provider_name = get_llm(force_provider)

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
        text = raw.strip()
        # Lớp 1+2: strip markdown
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

        try:
            data = json.loads(text)
            return {
                "answer":         data.get("answer", "Không có câu trả lời."),
                "source":         data.get("source", "Không xác định"),
                "confidence":     data.get("confidence", "low"),
                "related_topics": data.get("related_topics", []),
            }
        except json.JSONDecodeError:
            pass

        # Lớp 3: regex
        m = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return {
                    "answer":         data.get("answer", text),
                    "source":         data.get("source", "Không xác định"),
                    "confidence":     data.get("confidence", "medium"),
                    "related_topics": data.get("related_topics", []),
                }
            except json.JSONDecodeError:
                pass

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
          3. LLM call      → rate limit + retry backoff + fallback chain
          4. Parse JSON    → lưu cache
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

        # ── Bước 3: LLM Call ──
        context     = self._format_context(retrieved_docs)
        full_prompt = SYSTEM_PROMPT.format(context=context, question=question)
        raw_answer  = None

        # Fallback: Gemini (3 retry) → Groq (1 retry) → Ollama
        for provider_key, max_retries in [("gemini", 3), ("groq", 1), ("ollama", 0)]:
            try:
                self.llm, self.provider_name = get_llm(force_provider=provider_key)
            except Exception as e:
                print(f"[RAG] Không khởi tạo được {provider_key}: {str(e)[:60]}")
                continue

            success = False
            for attempt in range(max_retries + 1):
                try:
                    if provider_key == "gemini" and _llm_limiter:
                        with _llm_limiter:
                            response = self.llm.invoke([HumanMessage(content=full_prompt)])
                    else:
                        response = self.llm.invoke([HumanMessage(content=full_prompt)])

                    raw_answer = response.content
                    if _counter:
                        _counter.increment("gemini_llm" if provider_key == "gemini" else "groq")
                    success = True
                    break

                except Exception as e:
                    err   = str(e)
                    is_429 = any(k in err for k in [
                        "429", "quota", "ResourceExhausted", "RESOURCE_EXHAUSTED",
                    ])
                    if is_429 and attempt < max_retries:
                        delay = 2.0 * (2 ** attempt)  # 2, 4, 8 giây
                        print(f"[RAG] {self.provider_name} 429, retry sau {delay:.0f}s "
                              f"({attempt+1}/{max_retries})...")
                        time.sleep(delay)
                    elif is_429:
                        print(f"[RAG] {self.provider_name} hết quota → fallback tiếp theo")
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

        # ── Bước 4: Parse + Cache ──
        parsed               = self._parse_json_response(raw_answer)
        parsed["provider"]   = self.provider_name
        parsed["from_cache"] = False
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
        self.llm, self.provider_name = get_llm(force_provider=provider)


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