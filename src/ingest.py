# =============================================================
# ingest.py - NẠP VÀ INDEX TÀI LIỆU VÀO CHROMADB
# =============================================================
# Luồng xử lý (RAG Pipeline - bước 1):
#   PDF/DOCX → Load → Chunk → Embed (Gemini) → Lưu ChromaDB
#
# Hỗ trợ 2 loại file:
#   1. File KB có cấu trúc (KB001, KB002...): tách theo KB block
#   2. File PDF thường (sổ tay gốc SGU, scan...): tách theo ký tự
#
# Lý do chọn các thông số:
#   - chunk_size=1000: Đủ ngữ cảnh cho 1 câu trả lời
#   - chunk_overlap=200: Tránh mất thông tin ở ranh giới
#   - top_k=4: Giới hạn token đưa vào LLM (Token Management)
# =============================================================

import os
import sys
import glob
from pathlib import Path
from dotenv import load_dotenv

# LangChain - đọc tài liệu
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding: Gọi thẳng Gemini REST API v1 qua requests
import requests
from langchain.embeddings.base import Embeddings

# LangChain - Vector store local
from langchain_chroma import Chroma

load_dotenv()

# =============================================================
# CẤU HÌNH
# =============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
COLLECTION_NAME = "sgu_knowledge_base"


def load_documents(data_dir: str) -> list:
    """
    Đọc tất cả file PDF và DOCX trong thư mục data/.
    """
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"[!] Tạo thư mục {data_dir}...")
        data_path.mkdir(parents=True)
        print(f"[!] Hãy đặt file PDF/DOCX vào thư mục '{data_dir}' rồi chạy lại.")
        return []

    # --- Đọc file PDF ---
    pdf_files = list(data_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        print(f"[+] Đang đọc PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = pdf_file.name
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1
            documents.extend(docs)
            print(f"    → Đọc được {len(docs)} trang")
        except Exception as e:
            print(f"    [X] Lỗi khi đọc {pdf_file.name}: {e}")

    # --- Đọc file DOCX ---
    docx_files = list(data_path.glob("*.docx"))
    for docx_file in docx_files:
        print(f"[+] Đang đọc DOCX: {docx_file.name}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = docx_file.name
                doc.metadata["page"] = 1
            documents.extend(docs)
            print(f"    → Đọc được {len(docs)} document")
        except Exception as e:
            print(f"    [X] Lỗi khi đọc {docx_file.name}: {e}")

    print(f"\n[✓] Tổng cộng: {len(documents)} trang/document từ {len(pdf_files)+len(docx_files)} file")
    return documents


def split_documents(documents: list) -> list:
    """
    Tách tài liệu thành chunks chất lượng cao.

    Chiến lược:
    ─────────────────────────────────────────────────────────
    BƯỚC 1 – Nhận diện loại file:
        • File KB (cam_nang_sinh_vien.pdf): Có chứa pattern KB###
          → Tách theo KB block để giữ nguyên từng đơn vị kiến thức
        • File thường (so_tay_sv.pdf, scan,...): Không có pattern KB
          → Tách theo ký tự với chunk_size lớn hơn để giữ ngữ cảnh

    BƯỚC 2 – Xử lý từng loại:
        • KB file: Tìm pattern KB\d{3}[A-Z]? → mỗi block = 1 chunk
        • Thường file: Dùng RecursiveCharacterTextSplitter
          chunk_size=1000, overlap=200

    BƯỚC 3 – Lọc trang rác CHỈ trong KB file:
        • Chỉ skip trang bìa/mục lục của file KB structured
        • KHÔNG skip trang nào của file PDF thường
    ─────────────────────────────────────────────────────────
    """
    import re
    from langchain.schema import Document

    kb_chunks = []    # Chunks từ file có cấu trúc KB
    other_docs = []   # Trang từ file PDF thường (không có KB)

    # Keywords chỉ xuất hiện trong TRANG BÌA/MỤC LỤC của file KB structured
    # KHÔNG dùng để lọc file PDF thường
    KB_FILE_JUNK = [
        "KNOWLEDGE BASE",
        "Chunking Schema",
        "MỤC LỤC TOPICS",
        "Topic / Chủ đề",
        "Tổng số blocks",
        "Phạm vi:",
        '"id": "KB',           # JSON schema trong trang bìa
        "SGU RAG Chatbot",     # Header của file KB mình tạo
    ]

    # ──────────────────────────────────────────────────
    # Phân loại từng trang/document theo file nguồn
    # ──────────────────────────────────────────────────
    # Nhóm documents theo filename để xử lý theo từng file
    from collections import defaultdict
    file_docs = defaultdict(list)
    for doc in documents:
        fname = doc.metadata.get("filename", "unknown")
        file_docs[fname].append(doc)

    for filename, docs in file_docs.items():
        # Kiểm tra xem file này có phải file KB structured không
        # Bằng cách xem thử có trang nào chứa pattern KB### không
        all_text_sample = " ".join(d.page_content[:500] for d in docs[:5])
        is_kb_file = bool(re.search(r'KB\d{3}', all_text_sample))

        print(f"    → File '{filename}': {'KB structured' if is_kb_file else 'PDF thường'}")

        for doc in docs:
            text = doc.page_content

            # Bỏ qua trang trống hoàn toàn (< 30 ký tự)
            if len(text.strip()) < 30:
                page = doc.metadata.get("page", "?")
                print(f"    [skip] Trang {page} của '{filename}': nội dung rỗng/quá ngắn")
                continue

            if is_kb_file:
                # ── File KB structured: lọc trang bìa/mục lục rác ──
                is_junk = any(kw in text for kw in KB_FILE_JUNK)
                if is_junk:
                    page = doc.metadata.get("page", "?")
                    print(f"    [skip] Trang {page} của '{filename}': trang bìa/mục lục KB")
                    continue

                # Tìm KB blocks trong trang
                kb_matches = list(re.finditer(
                    r'(KB\d{3}[A-Z]?)\s+(.+?)(?=KB\d{3}[A-Z]?\s|\Z)',
                    text, re.DOTALL
                ))

                if kb_matches:
                    for match in kb_matches:
                        kb_id = match.group(1)
                        kb_content = match.group(0).strip()
                        if len(kb_content) < 50:
                            continue
                        kb_chunks.append(Document(
                            page_content=kb_content,
                            metadata={
                                **doc.metadata,
                                "kb_id": kb_id,
                                "chunk_type": "kb_block",
                            }
                        ))
                else:
                    # Trang KB file nhưng không có block → dùng splitter thường
                    other_docs.append(doc)
            else:
                # ── File PDF thường: KHÔNG lọc, đưa thẳng vào splitter ──
                other_docs.append(doc)

    # ──────────────────────────────────────────────────
    # Tách các trang "thường" bằng RecursiveCharacterTextSplitter
    # ──────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "。", " ", ""],
        length_function=len,
    )
    other_chunks = splitter.split_documents(other_docs) if other_docs else []

    # Đánh dấu chunk type cho other_chunks
    for chunk in other_chunks:
        chunk.metadata["chunk_type"] = "text_split"

    all_chunks = kb_chunks + other_chunks

    print(f"\n[✓] Tách thành {len(all_chunks)} chunks")
    print(f"    → KB blocks : {len(kb_chunks)} chunks (tách theo cấu trúc KB)")
    print(f"    → Text split: {len(other_chunks)} chunks (tách theo ký tự, chunk_size=1000)")
    return all_chunks


class GeminiEmbeddings(Embeddings):
    """
    Custom Embedding class gọi thẳng Gemini REST API v1beta qua requests.
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    def _embed(self, text: str, task_type: str) -> list:
        response = requests.post(
            self.url,
            params={"key": self.api_key},
            json={
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]["values"]

    def embed_documents(self, texts: list) -> list:
        embeddings = []
        for i, text in enumerate(texts):
            vec = self._embed(text, task_type="retrieval_document")
            embeddings.append(vec)
            if (i + 1) % 10 == 0:
                print(f"    → Đã embed {i+1}/{len(texts)} chunks...")
        return embeddings

    def embed_query(self, text: str) -> list:
        return self._embed(text, task_type="retrieval_query")


def create_embeddings():
    """Khởi tạo GeminiEmbeddings."""
    if not GOOGLE_API_KEY:
        raise ValueError("[X] Thiếu GOOGLE_API_KEY trong file .env!")

    embeddings = GeminiEmbeddings(
        api_key=GOOGLE_API_KEY,
        model="gemini-embedding-001",
    )
    print("[✓] Khởi tạo Gemini Embedding (gemini-embedding-001, v1beta)")
    return embeddings


def build_vector_store(chunks: list, embeddings) -> Chroma:
    """Tạo hoặc cập nhật ChromaDB với các chunks đã embed."""
    print(f"[+] Đang tạo vector store tại: {CHROMA_DB_PATH}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
    )
    print(f"[✓] Đã lưu {len(chunks)} chunks vào ChromaDB tại '{CHROMA_DB_PATH}'")
    return vector_store


def load_vector_store(embeddings) -> Chroma:
    """Load ChromaDB đã có sẵn từ disk."""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    return vector_store


def check_db_exists() -> bool:
    """Kiểm tra ChromaDB đã được tạo chưa."""
    db_path = Path(CHROMA_DB_PATH)
    return (db_path / "chroma.sqlite3").exists()


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
        ctype = c.metadata.get("chunk_type", "?")
        kid   = c.metadata.get("kb_id", "—")
        fname = c.metadata.get("filename", "?")
        page  = c.metadata.get("page", "?")
        print(f"    [{i+1}] type={ctype} | kb_id={kid} | file={fname} | page={page}")
        print(f"         preview: {c.page_content[:120].replace(chr(10),' ')}...")

    print("\n🔗 BƯỚC 3: KHỞI TẠO EMBEDDING")
    embeddings = create_embeddings()

    print("\n💾 BƯỚC 4: LƯU VÀO CHROMADB")
    vector_store = build_vector_store(chunks, embeddings)

    print("\n" + "=" * 60)
    print("  ✅ HOÀN TẤT! Có thể chạy app: streamlit run app.py")
    print("=" * 60)