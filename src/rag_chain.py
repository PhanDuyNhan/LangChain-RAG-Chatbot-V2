# =============================================================
# rag_chain.py - RAG PIPELINE CHÍNH (LEVEL 1)
# =============================================================
# Luồng xử lý:
#   Câu hỏi → Embed query → ChromaDB search (top_k=4)
#            → Ghép context → System Prompt + Context + Câu hỏi
#            → LLM → Parse JSON → Trả về có citations
#
# Structured Output:
#   Tất cả câu trả lời trả về JSON chuẩn:
#   {
#     "answer": "Câu trả lời tiếng Việt...",
#     "source": "Trang X, tài liệu Y",
#     "confidence": "high/medium/low",
#     "related_topics": ["topic1", "topic2"]
#   }
#
# Lý do dùng JSON thay vì văn bản tự do:
# - Dễ parse và hiển thị riêng từng phần trên UI
# - Buộc LLM cấu trúc hóa câu trả lời (ít "hallucination" hơn)
# - Dễ kiểm tra tự động (unit test)
# =============================================================

import os
import json
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Import router và ingest helper
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_router import get_llm, get_current_provider
from ingest import (
    COLLECTION_NAME, CHROMA_DB_PATH,
    check_db_exists, load_vector_store, create_embeddings
)

load_dotenv()

# =============================================================
# SYSTEM PROMPT - Trái tim của RAG Chatbot
# =============================================================
# Lý do thiết kế System Prompt này:
#
# 1. VAI TRÒ RÕ RÀNG: "trợ lý hỗ trợ sinh viên SGU" → LLM biết ngữ cảnh,
#    tránh trả lời lạc đề
#
# 2. BUỘC DÙNG CONTEXT: "CHỈ dựa vào thông tin được cung cấp" → Ngăn LLM
#    "hallucinate" thông tin không có trong tài liệu
#
# 3. STRUCTURED OUTPUT: Định nghĩa schema JSON rõ ràng với ví dụ cụ thể
#    → LLM ít bị lỗi format hơn
#
# 4. CITATIONS BẮT BUỘC: "source" phải chứa số trang → Sinh viên có thể
#    verify thông tin gốc
#
# 5. CONFIDENCE LEVEL: Giúp người dùng biết độ tin cậy của câu trả lời
#
# 6. FALLBACK KHI KHÔNG CÓ THÔNG TIN: Tránh LLM bịa đặt khi không tìm thấy
# =============================================================

SYSTEM_PROMPT = """Bạn là trợ lý hỗ trợ sinh viên của Trường Đại học Sài Gòn (SGU).

NHIỆM VỤ: Trả lời câu hỏi về quy định học vụ, thủ tục, học phí, học bổng của SGU.

NGUYÊN TẮC:
1. CHỈ dùng thông tin trong [CONTEXT] bên dưới.
2. KHÔNG bịa đặt. Nếu không có thông tin, trả lời: "Tôi không tìm thấy thông tin này."
3. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu.

⚠️ QUAN TRỌNG - FORMAT OUTPUT:
Chỉ trả về một object JSON duy nhất. KHÔNG viết gì khác ngoài JSON.
KHÔNG dùng markdown. KHÔNG dùng ```json. Chỉ trả về JSON thuần túy.

Schema bắt buộc:
{{"answer":"nội dung trả lời","source":"Trang X, tên file","confidence":"high","related_topics":["topic1"]}}

Ví dụ câu trả lời hợp lệ:
{{"answer":"Điều kiện tốt nghiệp gồm: (1) Không bị truy cứu hình sự. (2) Tích lũy đủ học phần. (3) GPA tích lũy từ 2.0 trở lên. (4) Có chứng chỉ GDQP và hoàn thành GDTC.","source":"Trang 18, cam_nang_sinh_vien.pdf","confidence":"high","related_topics":["điều kiện tốt nghiệp","GPA","học vụ"]}}

Giá trị confidence:
- "high": thông tin rõ ràng trong context
- "medium": thông tin không hoàn toàn khớp
- "low": suy luận gián tiếp

[CONTEXT]
{context}

[CÂU HỎI]
{question}

Trả lời (chỉ JSON, không gì khác):"""


# =============================================================
# CLASS RAGChain - Đóng gói toàn bộ pipeline RAG
# =============================================================

class RAGChain:
    """
    Pipeline RAG hoàn chỉnh:
      1. Nhận câu hỏi
      2. Tìm kiếm context liên quan từ ChromaDB (Vector Search)
      3. Ghép context vào prompt (Context Injection)
      4. Gọi LLM để sinh câu trả lời
      5. Parse JSON và trả về structured response
    """
    
    def __init__(self, force_provider: str = None):
        """
        Khởi tạo RAG Chain.
        
        Args:
            force_provider: Ép dùng provider cụ thể ('gemini'/'groq'/'ollama')
        """
        self.top_k = int(os.getenv("RETRIEVER_TOP_K", "4"))  # Token Management
        self._init_components(force_provider)
    
    def _init_components(self, force_provider: str = None):
        """Khởi tạo tất cả component: embedding, vector store, LLM."""
        # --- Embedding model ---
        self.embeddings = create_embeddings()
        
        # --- Vector Store ---
        if not check_db_exists():
            raise RuntimeError(
                "ChromaDB chưa được tạo!\n"
                "Hãy chạy: python src/ingest.py\n"
                "để nạp tài liệu trước."
            )
        self.vector_store = load_vector_store(self.embeddings)
        
        # --- Retriever: tìm top_k chunk liên quan nhất ---
        # top_k=4: Lấy 4 chunk, mỗi chunk ~1000 ký tự → tổng ~4000 ký tự context
        # Không quá nhiều để tiết kiệm token (Token Management)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",           # Tìm theo cosine similarity
            search_kwargs={"k": self.top_k}    # Giới hạn số chunk (Token Management)
        )
        
        # --- LLM (với fallback tự động) ---
        self.llm, self.provider_name = get_llm(force_provider)
    
    def _format_context(self, docs: list[Document]) -> str:
        """
        Định dạng danh sách Document thành chuỗi context để inject vào prompt.
        
        Mỗi chunk được đánh số và ghi rõ nguồn (trang, file) để LLM
        có thể trích dẫn chính xác trong phần "source" của JSON output.
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Lấy metadata nguồn
            filename = doc.metadata.get("filename", "Tài liệu SGU")
            page = doc.metadata.get("page", "?")
            
            # Format: [1] Trang 5, cam_nang_sinh_vien.pdf\n<nội dung>
            context_parts.append(
                f"[{i}] Trang {page}, {filename}\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _parse_json_response(self, raw_response: str) -> dict:
        """
        Parse JSON từ response của LLM với nhiều lớp fallback.

        Các trường hợp LLM có thể trả về:
        1. JSON thuần túy {"answer":...}               → parse trực tiếp
        2. JSON trong markdown ```json ... ```          → strip markdown rồi parse
        3. JSON lẫn trong text "Đây là câu trả lời: {" → dùng regex tìm JSON
        4. Plain text hoàn toàn (LLM không tuân thủ)   → dùng text làm answer

        Lý do cần nhiều lớp fallback:
        - Model nhỏ (Groq llama-3.1-8b) đôi khi không tuân thủ JSON format
        - Cần đảm bảo app không crash dù LLM trả về format bất kỳ
        """
        raw = raw_response.strip()

        # --- Lớp 1: Xóa markdown code block ---
        # Xử lý ```json ... ``` hoặc ``` ... ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        # --- Lớp 2: Parse JSON trực tiếp ---
        try:
            data = json.loads(raw)
            return {
                "answer":         data.get("answer", "Không có câu trả lời."),
                "source":         data.get("source", "Không xác định"),
                "confidence":     data.get("confidence", "low"),
                "related_topics": data.get("related_topics", []),
            }
        except json.JSONDecodeError:
            pass

        # --- Lớp 3: Tìm JSON object trong text bằng regex ---
        # Trường hợp LLM viết text trước rồi mới có JSON
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "answer":         data.get("answer", raw),
                    "source":         data.get("source", "Không xác định"),
                    "confidence":     data.get("confidence", "medium"),
                    "related_topics": data.get("related_topics", []),
                }
            except json.JSONDecodeError:
                pass

        # --- Lớp 4: Fallback hoàn toàn ---
        # LLM không trả về JSON → dùng toàn bộ text làm answer
        # Xóa các dòng dạng "[1] Trang X, file.pdf Nội dung..." (context bị lộ)
        lines = raw_response.strip().split("\n")
        clean_lines = [
            l for l in lines
            if not re.match(r"^\[\d+\]\s+Trang\s+\d+", l.strip())
        ]
        clean_answer = "\n".join(clean_lines).strip() or raw_response

        return {
            "answer":         clean_answer,
            "source":         "Xem mục Context bên dưới",
            "confidence":     "low",
            "related_topics": [],
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Hàm chính: Nhận câu hỏi, trả về JSON có cấu trúc.
        
        Luồng:
          1. Vector Search: Embed câu hỏi → Tìm top_k chunk tương tự
          2. Context Injection: Ghép các chunk thành context string
          3. LLM Call: Gọi LLM với system prompt + context + câu hỏi
          4. Parse & Return: Trả về dict có answer/source/confidence
        
        Args:
            question: Câu hỏi của sinh viên (tiếng Việt)
        
        Returns:
            {
              "answer": str,           # Câu trả lời
              "source": str,           # Trích dẫn nguồn (trang, tài liệu)
              "confidence": str,       # "high"/"medium"/"low"
              "related_topics": list,  # Chủ đề liên quan
              "provider": str,         # LLM đang dùng (cho UI)
              "retrieved_docs": list,  # Các chunk đã tìm thấy (debug)
            }
        """
        # --- Bước 1: Vector Search ---
        # Embed câu hỏi và tìm top_k chunk liên quan nhất
        retrieved_docs = self.retriever.invoke(question)
        
        if not retrieved_docs:
            return {
                "answer": "Không tìm thấy thông tin liên quan trong tài liệu. "
                          "Vui lòng liên hệ trực tiếp các phòng ban của trường SGU.",
                "source": "Không có",
                "confidence": "low",
                "related_topics": [],
                "provider": self.provider_name,
                "retrieved_docs": [],
            }
        
        # --- Bước 2: Context Injection ---
        context = self._format_context(retrieved_docs)
        
        # --- Bước 3: Gọi LLM (có tự động fallback khi 429) ---
        full_prompt = SYSTEM_PROMPT.format(context=context, question=question)
        from langchain.schema import HumanMessage

        raw_answer = None
        # Fallback theo thứ tự cố định: gemini → groq → ollama
        # Mỗi lần thử 1 provider, nếu 429 → chuyển sang provider KẾ TIẾP
        fallback_order = ["gemini", "groq", "ollama"]
        from llm_router import get_llm

        for i, provider_key in enumerate(fallback_order):
            # Khởi tạo LLM cho provider này
            try:
                self.llm, self.provider_name = get_llm(force_provider=provider_key)
            except Exception as e:
                print(f"[RAG] Không khởi tạo được {provider_key}: {str(e)[:60]}")
                continue  # Provider không khởi tạo được → thử tiếp

            # Gọi LLM
            try:
                response = self.llm.invoke([HumanMessage(content=full_prompt)])
                raw_answer = response.content
                break  # Thành công → dừng vòng lặp
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "ResourceExhausted" in err:
                    print(f"[RAG] {self.provider_name} hết quota → thử provider tiếp theo")
                    continue  # Thử provider kế tiếp
                else:
                    raise  # Lỗi khác → raise luôn, không fallback

        if raw_answer is None:
            return {
                "answer": "Tất cả LLM provider đều hết quota hoặc lỗi. "
                          "Vui lòng thử lại sau hoặc kiểm tra API key trong .env",
                "source": "Không có",
                "confidence": "low",
                "related_topics": [],
                "provider": self.provider_name,
                "retrieved_docs": [],
            }
        
        # --- Bước 4: Parse JSON ---
        parsed = self._parse_json_response(raw_answer)
        
        # Bổ sung metadata cho UI
        parsed["provider"] = self.provider_name
        parsed["retrieved_docs"] = [
            {
                "content": doc.page_content[:200] + "...",  # Rút gọn cho UI
                "page": doc.metadata.get("page", "?"),
                "filename": doc.metadata.get("filename", "?"),
            }
            for doc in retrieved_docs
        ]
        
        return parsed
    
    def switch_provider(self, provider: str):
        """
        Chuyển đổi LLM provider thủ công (người dùng chọn trên UI).
        
        Args:
            provider: 'gemini', 'groq', hoặc 'ollama'
        """
        print(f"[RAGChain] Chuyển sang provider: {provider}")
        self.llm, self.provider_name = get_llm(force_provider=provider)


# =============================================================
# SINGLETON PATTERN - Chỉ khởi tạo 1 lần để tái dùng embedding/DB
# =============================================================
_rag_chain_instance: Optional[RAGChain] = None

def get_rag_chain(force_provider: str = None) -> RAGChain:
    """
    Trả về RAGChain instance (singleton).
    
    Lý do dùng singleton:
    - ChromaDB và embedding model tốn thời gian khởi tạo
    - Streamlit re-run mỗi khi người dùng tương tác
    - Dùng st.cache_resource hoặc singleton để tránh init lại
    """
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = RAGChain(force_provider)
    elif force_provider:
        # Nếu force provider khác, switch
        _rag_chain_instance.switch_provider(force_provider)
    return _rag_chain_instance


# =============================================================
# TEST NHANH - chạy: python src/rag_chain.py
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST RAG CHAIN")
    print("=" * 60)
    
    chain = RAGChain()
    
    # Test câu hỏi mẫu về SGU
    test_questions = [
        "Thủ tục bảo lưu kết quả học tập gồm mấy bước?",
        "Học phí ngành Công nghệ thông tin năm 2025 là bao nhiêu?",
        "Điều kiện để được học bổng khuyến khích học tập?",
    ]
    
    for q in test_questions:
        print(f"\n❓ Câu hỏi: {q}")
        result = chain.query(q)
        print(f"💬 Trả lời: {result['answer'][:200]}...")
        print(f"📄 Nguồn: {result['source']}")
        print(f"📊 Độ tin cậy: {result['confidence']}")
        print(f"🤖 Provider: {result['provider']}")
