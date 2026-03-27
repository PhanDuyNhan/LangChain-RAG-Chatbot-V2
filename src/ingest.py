# =============================================================
# src/ingest.py - NẠP VÀ INDEX TÀI LIỆU VÀO CHROMADB
# =============================================================
# Luồng xử lý (RAG Pipeline - bước 1):
#   PDF/DOCX → Load → Chunk → Embed (Gemini) → Lưu ChromaDB
#
# Hỗ trợ 2 loại file:
#   1. File KB có cấu trúc (KB001, KB002...): tách theo KB block
#   2. File PDF thường (sổ tay gốc SGU, scan...): tách theo ký tự
#
# Tối ưu quota (đã cải thiện):
#   - RateLimiter 90 RPM cho Gemini Embedding API
#   - MIN DELAY 0.7s giữa mỗi embed call → tránh burst 429
#   - JITTER ngẫu nhiên 0–0.3s → tránh thundering herd
#   - Retry với exponential backoff khi 429
#   - Batch delay 2s sau mỗi 10 chunks (tăng từ 1s)
# =============================================================

import os
import sys
import time
import random
import functools
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma

# Import quota_guard từ thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from quota_guard import (
        get_embed_limiter,
        with_retry,
        get_counter,
        get_daily_tracker,
        GEMINI_EMBED_RPD_SOFT,
    )
    _embed_limiter = get_embed_limiter()
    _counter       = get_counter()
    _daily_tracker = get_daily_tracker()
    _with_retry    = with_retry
except ImportError:
    _embed_limiter = None
    _counter       = None
    _daily_tracker = None
    def _with_retry(max_retries=3, base_delay=2.0):
        def decorator(func): return func
        return decorator

load_dotenv()

# =============================================================
# CẤU HÌNH
# =============================================================
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_DIR        = os.getenv("DATA_DIR", "./data")
COLLECTION_NAME = "sgu_knowledge_base"

# Delay tối thiểu giữa các embed call (giây)
# 90 RPM = 1.5 req/giây → min delay 0.7s để an toàn
EMBED_MIN_DELAY  = float(os.getenv("EMBED_MIN_DELAY", "0.7"))
EMBED_JITTER_MAX = 0.3   # jitter ngẫu nhiên thêm 0-0.3s


# =============================================================
# LOAD DOCUMENTS
# =============================================================
def load_documents(data_dir: str) -> list:
    """Đọc tất cả file PDF và DOCX trong thư mục data/."""
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"[!] Tạo thư mục {data_dir}...")
        data_path.mkdir(parents=True)
        print(f"[!] Hãy đặt file PDF/DOCX vào thư mục '{data_dir}' rồi chạy lại.")
        return []

    pdf_files  = list(data_path.glob("*.pdf"))
    docx_files = list(data_path.glob("*.docx"))

    for pdf_file in pdf_files:
        print(f"[+] Đang đọc PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = pdf_file.name
                doc.metadata["page"]     = doc.metadata.get("page", 0) + 1
            documents.extend(docs)
            print(f"    → Đọc được {len(docs)} trang")
        except Exception as e:
            print(f"    [X] Lỗi khi đọc {pdf_file.name}: {e}")

    for docx_file in docx_files:
        print(f"[+] Đang đọc DOCX: {docx_file.name}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = docx_file.name
                doc.metadata["page"]     = 1
            documents.extend(docs)
            print(f"    → Đọc được {len(docs)} document")
        except Exception as e:
            print(f"    [X] Lỗi khi đọc {docx_file.name}: {e}")

    print(f"\n[✓] Tổng cộng: {len(documents)} trang/document từ {len(pdf_files)+len(docx_files)} file")
    return documents


# =============================================================
# SPLIT DOCUMENTS
# =============================================================
def split_documents(documents: list) -> list:
    """
    Tách tài liệu thành chunks chất lượng cao.

    Chiến lược:
      • File KB (có pattern KB###): tách theo KB block → giữ nguyên đơn vị kiến thức
      • File thường (PDF thường):   RecursiveCharacterTextSplitter chunk_size=1000
    """
    import re
    from langchain.schema import Document
    from collections import defaultdict

    KB_FILE_JUNK = [
        "KNOWLEDGE BASE", "Chunking Schema", "MỤC LỤC TOPICS",
        "Topic / Chủ đề", "Tổng số blocks", "Phạm vi:",
        '"id": "KB', "SGU RAG Chatbot",
    ]

    kb_chunks  = []
    other_docs = []

    file_docs: dict = defaultdict(list)
    for doc in documents:
        file_docs[doc.metadata.get("filename", "unknown")].append(doc)

    for filename, docs in file_docs.items():
        sample        = " ".join(d.page_content[:500] for d in docs[:5])
        is_kb_file    = bool(re.search(r'KB\d{3}', sample))
        print(f"    → File '{filename}': {'KB structured' if is_kb_file else 'PDF thường'}")

        for doc in docs:
            text = doc.page_content

            if len(text.strip()) < 30:
                continue

            if is_kb_file:
                if any(kw in text for kw in KB_FILE_JUNK):
                    continue

                kb_matches = list(re.finditer(
                    r'(KB\d{3}[A-Z]?)\s+(.+?)(?=KB\d{3}[A-Z]?\s|\Z)',
                    text, re.DOTALL,
                ))
                if kb_matches:
                    for match in kb_matches:
                        kb_content = match.group(0).strip()
                        if len(kb_content) < 50:
                            continue
                        kb_chunks.append(Document(
                            page_content=kb_content,
                            metadata={**doc.metadata, "kb_id": match.group(1), "chunk_type": "kb_block"},
                        ))
                else:
                    other_docs.append(doc)
            else:
                other_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", "。", " ", ""],
        length_function=len,
    )
    other_chunks = splitter.split_documents(other_docs) if other_docs else []
    for chunk in other_chunks:
        chunk.metadata["chunk_type"] = "text_split"

    all_chunks = kb_chunks + other_chunks
    print(f"\n[✓] Tách thành {len(all_chunks)} chunks")
    print(f"    → KB blocks : {len(kb_chunks)}")
    print(f"    → Text split: {len(other_chunks)}")
    return all_chunks


# =============================================================
# GEMINI EMBEDDINGS (với Rate Limit + Retry + Min Delay)
# =============================================================
class GeminiEmbeddings(Embeddings):
    """
    Gọi Gemini Embedding REST API với:
      - RateLimiter 90 RPM    → không bị 429 khi embed nhiều chunks
      - Min delay 0.7s/call   → tránh burst ngắn gây 429 tức thì
      - Jitter 0–0.3s         → tránh thundering herd
      - Retry exponential backoff khi 429 xảy ra
      - Batch delay 2s sau mỗi 10 chunks → tránh sustained burst
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model   = model
        self.url     = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:embedContent"
        )
        self._last_call_time = 0.0  # Theo dõi thời điểm gọi cuối

    def _throttle(self):
        """Đảm bảo khoảng cách tối thiểu giữa các API call."""
        now     = time.time()
        elapsed = now - self._last_call_time
        min_gap = EMBED_MIN_DELAY + random.uniform(0, EMBED_JITTER_MAX)

        if elapsed < min_gap:
            wait = min_gap - elapsed
            time.sleep(wait)

        self._last_call_time = time.time()

    def _embed_single(self, text: str, task_type: str) -> list:
        """Gọi API 1 text, có throttle + retry backoff."""

        @_with_retry(max_retries=4, base_delay=3.0)
        def _call():
            resp = requests.post(
                self.url,
                params={"key": self.api_key},
                json={"content": {"parts": [{"text": text}]}, "taskType": task_type},
                timeout=30,
            )
            if resp.status_code == 429:
                # Raise để trigger retry backoff
                raise Exception(f"429 Too Many Requests: {resp.text[:100]}")
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]

        # Throttle trước khi gọi
        self._throttle()

        if _daily_tracker and not _daily_tracker.can_consume("gemini_embed", GEMINI_EMBED_RPD_SOFT):
            raise RuntimeError(
                "Đã chạm soft cap embedding Gemini trong ngày. "
                "Hãy tái sử dụng ChromaDB hiện có hoặc chờ quota ngày mới."
            )

        if _embed_limiter:
            with _embed_limiter:
                result = _call()
        else:
            result = _call()

        if _counter:
            _counter.increment("gemini_embed")
        if _daily_tracker:
            _daily_tracker.increment("gemini_embed")

        return result

    def embed_documents(self, texts: list) -> list:
        embeddings = []
        total      = len(texts)
        for i, text in enumerate(texts):
            vec = self._embed_single(text, task_type="retrieval_document")
            embeddings.append(vec)

            # Progress log
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"    → Đã embed {i+1}/{total} chunks...")

            # Batch delay sau mỗi 10 chunks (tăng 1s → 2s để an toàn)
            if (i + 1) % 10 == 0 and (i + 1) < total:
                print(f"    → Nghỉ 2s tránh quota burst...")
                time.sleep(2.0)

        return embeddings

    def embed_query(self, text: str) -> list:
        return self._embed_single(text, task_type="retrieval_query")


# =============================================================
# HELPERS
# =============================================================
def create_embeddings() -> GeminiEmbeddings:
    if not GOOGLE_API_KEY:
        raise ValueError("[X] Thiếu GOOGLE_API_KEY trong file .env!")
    print("[✓] Khởi tạo Gemini Embedding (gemini-embedding-001)")
    return GeminiEmbeddings(api_key=GOOGLE_API_KEY)


def build_vector_store(chunks: list, embeddings) -> Chroma:
    print(f"[+] Đang tạo vector store tại: {CHROMA_DB_PATH}")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
    )
    print(f"[✓] Đã lưu {len(chunks)} chunks vào ChromaDB")
    return vs


def load_vector_store(embeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )


def check_db_exists() -> bool:
    return (Path(CHROMA_DB_PATH) / "chroma.sqlite3").exists()


# =============================================================
# CHẠY TRỰC TIẾP: python src/ingest.py
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG CHATBOT SGU - INGEST TÀI LIỆU")
    print("=" * 60)

    print("\n📂 BƯỚC 1: ĐỌC TÀI LIỆU")
    documents = load_documents(DATA_DIR)
    if not documents:
        print("\n[!] Không có tài liệu nào. Kết thúc.")
        sys.exit(1)

    print("\n✂️  BƯỚC 2: TÁCH CHUNKS")
    chunks = split_documents(documents)

    print(f"\n    Xem thử 3 chunks đầu:")
    for i, c in enumerate(chunks[:3]):
        print(f"    [{i+1}] type={c.metadata.get('chunk_type','?')} | "
              f"kb_id={c.metadata.get('kb_id','—')} | "
              f"file={c.metadata.get('filename','?')} | "
              f"page={c.metadata.get('page','?')}")
        print(f"         preview: {c.page_content[:120].replace(chr(10),' ')}...")

    print("\n🔗 BƯỚC 3: KHỞI TẠO EMBEDDING")
    embeddings = create_embeddings()

    print("\n💾 BƯỚC 4: LƯU VÀO CHROMADB")
    build_vector_store(chunks, embeddings)

    print("\n" + "=" * 60)
    print("  ✅ HOÀN TẤT! Có thể chạy app: streamlit run app.py")
    print("=" * 60)
