# =============================================================
# ingest.py - NẠP VÀ INDEX TÀI LIỆU VÀO CHROMADB
# =============================================================
# Luồng xử lý (RAG Pipeline - bước 1):
#   PDF/DOCX → Load → Chunk → Embed (Gemini) → Lưu ChromaDB
#
# Lý do chọn các thông số:
#   - chunk_size=1000: Đủ ngữ cảnh cho 1 câu trả lời, không quá dài
#   - chunk_overlap=200: Tránh mất thông tin ở ranh giới giữa 2 chunk
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
# Lý do bỏ google.generativeai và langchain-google-genai:
#   - Cả hai đều hardcode dùng google-ai-generativelanguage v1beta bên dưới
#   - embedding model (text-embedding-004) chỉ có trên REST API v1
#   - Không có cách nào override được qua SDK
# Giải pháp: Dùng requests gọi thẳng endpoint v1, không qua SDK nào
#   URL: https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent
import requests                                    # Gọi HTTP request trực tiếp
from langchain.embeddings.base import Embeddings   # Base class để ChromaDB nhận ra

# LangChain - Vector store local (không cần server)
from langchain_chroma import Chroma

# Nạp biến môi trường từ file .env
load_dotenv()

# =============================================================
# CẤU HÌNH - Tất cả lấy từ .env để dễ thay đổi
# =============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
COLLECTION_NAME = "sgu_knowledge_base"  # Tên collection trong ChromaDB


def load_documents(data_dir: str) -> list:
    """
    Đọc tất cả file PDF và DOCX trong thư mục data/.
    
    Trả về: List các Document của LangChain, mỗi Document có:
        - page_content: Nội dung văn bản
        - metadata: {'source': tên file, 'page': số trang}
    """
    documents = []
    data_path = Path(data_dir)
    
    # Kiểm tra thư mục tồn tại
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
            # PyPDFLoader tự động tách theo trang, thêm metadata page number
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Bổ sung metadata tên file (để dùng cho citations)
            for doc in docs:
                doc.metadata["filename"] = pdf_file.name
                # page bắt đầu từ 0, cộng 1 cho dễ đọc
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
                doc.metadata["page"] = 1  # DOCX không có số trang tự nhiên
            documents.extend(docs)
            print(f"    → Đọc được {len(docs)} document")
        except Exception as e:
            print(f"    [X] Lỗi khi đọc {docx_file.name}: {e}")

    print(f"\n[✓] Tổng cộng: {len(documents)} trang/document từ {len(pdf_files)+len(docx_files)} file")
    return documents


def split_documents(documents: list) -> list:
    """
    Tách tài liệu thành chunks chất lượng cao.

    Chiến lược 2 lớp cho file cam_nang_sinh_vien.pdf:

    Lớp 1 - Tách theo KB block (KB001, KB002...):
    - File PDF đã có cấu trúc knowledge base sẵn, mỗi KB là 1 chủ đề độc lập
    - Tách theo pattern "KB\d+" giữ nguyên toàn bộ 1 KB trong 1 chunk
    - Tránh gộp KB không liên quan vào chung → LLM không bị nhiễu
    - Ví dụ: KB006 (bảo lưu) và KB012 (học phí) sẽ KHÔNG nằm chung 1 chunk

    Lớp 2 - Fallback RecursiveCharacterTextSplitter:
    - Dùng cho các trang không có cấu trúc KB (trang mục lục, trang bìa...)
    - chunk_size=600: nhỏ hơn trước để mỗi chunk tập trung hơn
    - chunk_overlap=100: đủ để không mất context ở biên

    Kết quả: chunk chất lượng cao hơn → retrieval chính xác hơn → câu trả lời đúng hơn
    """
    import re
    from langchain.schema import Document

    kb_chunks = []       # Chunks tách theo KB block
    other_docs = []      # Trang không có KB block → dùng splitter thường

    # Các từ khóa nhận dạng trang RÁC — không có thông tin thực
    # Trang 1: metadata file (KNOWLEDGE BASE, Chunking Schema...)
    # Trang 2-3: mục lục (MỤC LỤC TOPICS, STT, Topic / Chủ đề...)
    JUNK_KEYWORDS = [
        "KNOWLEDGE BASE", "Chunking Schema", "MỤC LỤC TOPICS",
        "Topic / Chủ đề", "Tổng số blocks", "Phạm vi:"
    ]

    for doc in documents:
        text = doc.page_content

        # --- Lọc trang rác: mục lục, metadata, trang bìa ---
        # Nếu trang chứa từ khóa rác → bỏ qua hoàn toàn, không chunk
        is_junk = any(kw in text for kw in JUNK_KEYWORDS)
        if is_junk:
            page = doc.metadata.get("page", "?")
            print(f"    [skip] Bỏ trang {page} (mục lục/metadata không có giá trị)")
            continue

        # Tìm tất cả KB block trong trang: KB001, KB002, KB028A, KB028B...
        # Pattern: KB + 3 chữ số + tuỳ chọn 1 chữ cái (KB028A, KB028B...)
        kb_matches = list(re.finditer(r'(KB\d{3}[A-Z]?)\s+(.+?)(?=KB\d{3}[A-Z]?\s|\Z)', text, re.DOTALL))

        if kb_matches:
            # Trang có KB blocks → tách từng KB thành 1 chunk riêng
            for match in kb_matches:
                kb_id = match.group(1)               # Ví dụ: "KB006"
                kb_content = match.group(0).strip()  # Toàn bộ nội dung KB

                # Bỏ qua KB quá ngắn (< 50 ký tự) — thường là header thừa
                if len(kb_content) < 50:
                    continue

                # Tạo Document mới với metadata đầy đủ
                kb_chunks.append(Document(
                    page_content=kb_content,
                    metadata={
                        **doc.metadata,           # Giữ metadata gốc (page, filename)
                        "kb_id": kb_id,           # Thêm KB ID để dễ trace
                        "chunk_type": "kb_block", # Đánh dấu loại chunk
                    }
                ))
        else:
            # Trang không có KB → dùng splitter thông thường
            other_docs.append(doc)

    # Splitter cho các trang không có cấu trúc KB
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,    # Nhỏ hơn 1000 → mỗi chunk tập trung hơn, ít nhiễu hơn
        chunk_overlap=100, # Đủ để không mất context ở biên
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    other_chunks = splitter.split_documents(other_docs) if other_docs else []

    # Gộp 2 loại chunk lại
    all_chunks = kb_chunks + other_chunks

    print(f"[✓] Tách thành {len(all_chunks)} chunks")
    print(f"    → KB blocks: {len(kb_chunks)} chunks (tách theo cấu trúc KB)")
    print(f"    → Other:     {len(other_chunks)} chunks (tách theo ký tự)")
    return all_chunks


class GeminiEmbeddings(Embeddings):
    """
    Custom Embedding class gọi thẳng Gemini REST API v1 qua requests.

    Tại sao dùng requests thay vì SDK:
    - google.generativeai (cũ) và langchain-google-genai đều dùng
      google-ai-generativelanguage v1beta bên dưới → embedding 404
    - Không có cách override endpoint trong các SDK này
    - Giải pháp dứt điểm: gọi thẳng REST endpoint v1:
      POST https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent
    - requests là thư viện HTTP chuẩn, không phụ thuộc SDK nào

    Token Management:
    - Mỗi lần embed gọi 1 HTTP request → chỉ chạy khi ingest (1 lần duy nhất)
    - Sau khi ingest xong, ChromaDB lưu vector xuống disk → không embed lại
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.api_key = api_key   # Gemini API key để xác thực
        self.model = model       # Tên model (không cần prefix "models/")
        # URL endpoint REST API v1 — không phải v1beta
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    def _embed(self, text: str, task_type: str) -> list:
        """
        Hàm nội bộ: gọi REST API v1 để embed 1 đoạn text.

        Args:
            text:      Đoạn văn bản cần embed
            task_type: "retrieval_document" (tài liệu) hoặc "retrieval_query" (câu hỏi)

        Returns:
            List float — vector embedding của text
        """
        # Gửi POST request đến Gemini REST API v1
        # KHÔNG truyền "model" trong body — model đã có trong URL rồi
        # Nếu truyền thêm "model" trong body → bị ghép thành "models/models/..." → 404
        response = requests.post(
            self.url,
            params={"key": self.api_key},   # API key truyền qua query param
            json={
                "content": {"parts": [{"text": text}]},  # Nội dung cần embed
                "taskType": task_type,        # Loại task để tối ưu vector
            },
            timeout=30,  # Timeout 30 giây tránh treo vô hạn
        )

        # Kiểm tra HTTP status code — raise exception nếu lỗi (4xx, 5xx)
        response.raise_for_status()

        # Parse JSON response và trả về vector embedding
        return response.json()["embedding"]["values"]

    def embed_documents(self, texts: list) -> list:
        """
        Embed list các đoạn văn bản (gọi khi ingest tài liệu vào ChromaDB).

        Dùng task_type="retrieval_document" → tối ưu cho văn bản được lưu trữ
        để tìm kiếm sau này.

        Args:
            texts: List các chuỗi văn bản cần embed

        Returns:
            List các vector float, mỗi vector ứng với 1 đoạn văn bản
        """
        embeddings = []
        for i, text in enumerate(texts):
            # Embed từng text một và thêm vào list kết quả
            vec = self._embed(text, task_type="retrieval_document")
            embeddings.append(vec)
            # Log tiến độ mỗi 10 chunk để theo dõi (48 chunks tổng)
            if (i + 1) % 10 == 0:
                print(f"    → Đã embed {i+1}/{len(texts)} chunks...")
        return embeddings

    def embed_query(self, text: str) -> list:
        """
        Embed câu hỏi của người dùng (gọi mỗi khi tìm kiếm trong ChromaDB).

        Dùng task_type="retrieval_query" — khác với embed_documents vì:
        - Vector câu hỏi và vector tài liệu được tối ưu để "khớp" nhau
        - Giúp cosine similarity chính xác hơn

        Args:
            text: Câu hỏi của người dùng

        Returns:
            Vector float đại diện cho câu hỏi
        """
        return self._embed(text, task_type="retrieval_query")


def create_embeddings():
    """
    Khởi tạo GeminiEmbeddings (gọi trực tiếp google.generativeai API v1).

    Token Management:
    - Embedding chỉ chạy 1 lần khi ingest, không tốn quota LLM
    - Kết quả lưu vào ChromaDB local, tái sử dụng mãi
    """
    if not GOOGLE_API_KEY:
        raise ValueError("[X] Thiếu GOOGLE_API_KEY trong file .env!")

    embeddings = GeminiEmbeddings(
        api_key=GOOGLE_API_KEY,
        model="gemini-embedding-001",  # Model xác nhận hoạt động với free tier API key
    )
    print("[✓] Khởi tạo Gemini Embedding (gemini-embedding-001, v1beta)")
    return embeddings


def build_vector_store(chunks: list, embeddings) -> Chroma:
    """
    Tạo hoặc cập nhật ChromaDB với các chunks đã embed.
    
    ChromaDB lưu local tại CHROMA_DB_PATH (mặc định ./chroma_db/).
    Lần sau không cần embed lại, chỉ load từ disk.
    
    Token Management:
    - Chỉ embed những chunk chưa có trong DB (tránh embed trùng)
    - Dữ liệu persist trên disk, tiết kiệm quota API
    """
    print(f"[+] Đang tạo vector store tại: {CHROMA_DB_PATH}")
    
    # Xóa DB cũ nếu đang rebuild (để tránh duplicate)
    # Dùng persist_directory để lưu xuống disk
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
    )
    
    print(f"[✓] Đã lưu {len(chunks)} chunks vào ChromaDB tại '{CHROMA_DB_PATH}'")
    return vector_store


def load_vector_store(embeddings) -> Chroma:
    """
    Load ChromaDB đã có sẵn từ disk (không cần embed lại).
    Dùng trong rag_chain.py để truy vấn nhanh.
    """
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    return vector_store


def check_db_exists() -> bool:
    """Kiểm tra ChromaDB đã được tạo chưa."""
    db_path = Path(CHROMA_DB_PATH)
    # ChromaDB tạo file chroma.sqlite3 khi có dữ liệu
    return (db_path / "chroma.sqlite3").exists()


# =============================================================
# CHẠY TRỰC TIẾP: python src/ingest.py
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG CHATBOT SGU - INGEST TÀI LIỆU")
    print("=" * 60)
    
    # Bước 1: Đọc tài liệu
    print("\n📂 BƯỚC 1: ĐỌC TÀI LIỆU")
    documents = load_documents(DATA_DIR)
    
    if not documents:
        print("\n[!] Không có tài liệu nào. Kết thúc.")
        sys.exit(1)
    
    # Bước 2: Tách chunk
    print("\n✂️  BƯỚC 2: TÁCH CHUNKS")
    chunks = split_documents(documents)
    
    # Bước 3: Tạo embedding model
    print("\n🔗 BƯỚC 3: KHỞI TẠO EMBEDDING")
    embeddings = create_embeddings()
    
    # Bước 4: Lưu vào ChromaDB
    print("\n💾 BƯỚC 4: LƯU VÀO CHROMADB")
    vector_store = build_vector_store(chunks, embeddings)
    
    print("\n" + "=" * 60)
    print("  ✅ HOÀN TẤT! Có thể chạy app: streamlit run app.py")
    print("=" * 60)