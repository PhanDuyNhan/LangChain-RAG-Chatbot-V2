"""
Demo Semantic Chunking — Bài tập kiểm tra thực hành (Chương 4, mục 4.1)

So sánh số lượng chunks khi áp dụng SemanticChunker với 2 cấu hình
breakpoint_threshold_amount khác nhau (95 vs 60) trên cùng 1 đoạn văn bản.

Cách chạy:
    .venv/bin/python src/semantic_chunking.py
    hoặc
    python src/semantic_chunking.py
"""
import os
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# =============================================================
# DỮ LIỆU ĐẦU VÀO (theo đề bài)
# =============================================================
SAMPLE_TEXT = (
    "LangChain là một framework mã nguồn mở giúp đơn giản hóa việc xây dựng "
    "các ứng dụng dựa trên mô hình ngôn ngữ lớn (LLM). Nó cung cấp các module "
    "để quản lý prompt, kết nối LLM, và xây dựng các chuỗi xử lý phức tạp. "
    "Một trong những ứng dụng phổ biến nhất là hệ thống Hỏi-Đáp Tăng cường "
    "Truy xuất (RAG). Hệ thống RAG kết hợp LLM với một cơ sở dữ liệu vector "
    "để cung cấp câu trả lời chính xác hơn. Trong một diễn biến khác, thị "
    "trường chứng khoán hôm nay có nhiều biến động. Chỉ số VN-Index giảm nhẹ "
    "vào cuối phiên giao dịch buổi sáng. Các nhà đầu tư đang tỏ ra thận "
    "trọng trước các thông tin vĩ mô."
)


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Khởi tạo Gemini Embedding (dùng GOOGLE_API_KEY trong .env)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu GOOGLE_API_KEY trong file .env")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def run_config(embeddings, threshold: int, label: str) -> int:
    """Chạy SemanticChunker với 1 ngưỡng, in kết quả, trả về số chunks."""
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"\n{'='*60}")
    print(f"  {label} — breakpoint_threshold_amount={threshold}")
    print(f"{'='*60}")
    print(f"  Số lượng chunks: {len(chunks)}\n")
    for i, c in enumerate(chunks, 1):
        preview = c.strip().replace("\n", " ")
        print(f"  [Chunk {i}] {preview[:120]}{'...' if len(preview) > 120 else ''}")
    return len(chunks)


def main() -> None:
    print("=" * 60)
    print("  DEMO SEMANTIC CHUNKING — So sánh ngưỡng 95 vs 60")
    print("=" * 60)

    embeddings = build_embeddings()

    n95 = run_config(embeddings, 95, "Cấu hình 1 (ngưỡng cao)")
    n60 = run_config(embeddings, 60, "Cấu hình 2 (ngưỡng thấp)")

    print(f"\n{'='*60}")
    print("  GIẢI THÍCH")
    print(f"{'='*60}")
    print(
        f"  Khi breakpoint_threshold_amount=95 ({n95} chunks): splitter\n"
        f"  chỉ ngắt tại những điểm có sự thay đổi ngữ nghĩa rất lớn\n"
        f"  (top 5% khác biệt), do đó tạo ra ít chunks hơn, mỗi chunk\n"
        f"  mạch lạc về một chủ đề.\n\n"
        f"  Khi ngưỡng giảm xuống 60 ({n60} chunks): splitter trở nên\n"
        f"  nhạy hơn và sẽ ngắt ở nhiều điểm hơn, tạo ra nhiều chunks\n"
        f"  ngắn hơn."
    )


if __name__ == "__main__":
    main()
