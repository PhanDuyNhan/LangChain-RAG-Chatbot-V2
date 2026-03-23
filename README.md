# 🎓 SGU Chatbot - RAG & Multimodal Hỗ trợ Sinh viên

**Môn học:** Các Công nghệ Lập trình Hiện đại  
**Stack:** LangChain + Gemini 2.0 Flash + ChromaDB + Streamlit  
**Chi phí:** MIỄN PHÍ HOÀN TOÀN  
**Cấp độ đạt:** Mức 1 (RAG) + Mức 2 (Multimodal)

---

## 📁 Cấu Trúc Dự Án

```
rag-chatbot-sgu/
├── data/                          # Đặt file PDF/DOCX tại đây
│   └── cam_nang_sinh_vien.pdf    # File tài liệu SGU
├── src/
│   ├── ingest.py         # [Mức 1] Load → Chunk → Embed → ChromaDB
│   ├── llm_router.py     # Token Management: Gemini → Groq → Ollama
│   ├── rag_chain.py      # [Mức 1] RAG Pipeline + Structured JSON Output
│   ├── multimodal.py     # [Mức 2] Visual Reasoning + Gemini File API
│   └── agent.py          # ReAct Agent + 4 custom tools
├── app.py                # Streamlit UI (2 tab)
├── .env                  # API keys (KHÔNG commit lên git!)
└── requirements.txt      # Thư viện Python
```

---

## 🚀 Hướng Dẫn Cài Đặt

### Bước 1: Cài Python packages

```bash
pip install -r requirements.txt
```

### Bước 2: Cài đặt API Keys trong file `.env`

Tạo file `.env` ở thư mục gốc với nội dung sau:

```
GOOGLE_API_KEY=AIza...       # Bắt buộc
GROQ_API_KEY=gsk_...         # Khuyến nghị (backup)
```

**Lấy Gemini API Key (bắt buộc):**
1. Truy cập: https://aistudio.google.com/app/apikey
2. Click "Create API Key" → Copy key
3. Dán vào `.env`: `GOOGLE_API_KEY=AIza...`

**Lấy Groq API Key (backup, khuyến nghị):**
1. Truy cập: https://console.groq.com/keys
2. Tạo key miễn phí → Copy
3. Dán vào `.env`: `GROQ_API_KEY=gsk_...`

### Bước 3 (Tùy chọn): Cài Ollama — LLM backup local

Ollama chạy hoàn toàn trên máy, không cần internet, không giới hạn request. Dùng khi Gemini và Groq hết quota.

**3.1 — Tải và cài đặt:**
1. Truy cập: https://ollama.com/download
2. Tải `OllamaSetup.exe` → cài đặt bình thường
3. Ollama tự chạy nền sau khi cài (icon ở system tray)

**3.2 — Download model:**
```cmd
ollama pull qwen2.5:3b
```
> Chờ 5–15 phút (model ~1.9GB). Xong khi hiện chữ `success`.

**3.3 — Kiểm tra hoạt động:**
```cmd
ollama run qwen2.5:3b
```
Gõ `Xin chào` → model trả lời được là OK. Thoát: `/bye`

**Yêu cầu phần cứng tối thiểu cho qwen2.5:3b:**

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|-------------|
| RAM | 8 GB | 16 GB |
| Dung lượng disk | 3 GB | 5 GB |
| CPU | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| GPU | Không bắt buộc | NVIDIA (nhanh hơn) |

### Bước 4: Nạp tài liệu vào ChromaDB

```bash
python src/ingest.py
```

Lần đầu chạy sẽ gọi Gemini Embedding API (~30–60 giây). Chỉ cần chạy **một lần** — dữ liệu được lưu vào `chroma_db/` và tái sử dụng.

### Bước 5: Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại: http://localhost:8501

---

## 🏗️ Kiến Trúc Kỹ Thuật

### Mức 1 — RAG Pipeline

```
PDF/DOCX → PyPDFLoader / Docx2txtLoader
         → RecursiveTextSplitter (chunk_size=1000, overlap=200)
         → GeminiEmbeddings (gemini-embedding-001)
         → ChromaDB (local persist)

Query → Embed → similarity_search(k=4) → Context
      → System Prompt + Context + Question → LLM
      → Parse JSON → {"answer", "source", "confidence", "related_topics"}
```

**Kỹ thuật Cookbook áp dụng:**
- *JSON Mode* — System prompt ép LLM trả về JSON thuần túy (không markdown)
- *RAG with ChromaDB* — Vector search với Gemini Embeddings
- *Chunking Strategy* — Tách KB block cho file có cấu trúc, RecursiveTextSplitter cho PDF thường

### Mức 2 — Multimodal

**Visual Reasoning (ảnh):**
```
Image → PIL resize (max 1024px) → base64
      → Gemini 2.0 Flash Vision + System Prompt suy luận
      → JSON: {image_type, extracted_data, reasoning, answer, recommendations}
```

**Gemini File API (video/audio):**
```
Video/Audio → genai.upload_file() → Google Server (48h)
            → Chờ state ACTIVE
            → Gemini đọc từ URI (không dùng base64)
            → JSON: {summary, action_items, key_moments}
            → genai.delete_file() dọn dẹp
```

**Kỹ thuật Cookbook áp dụng:**
- *File API (video_understanding)* — Upload file lớn lên server thay vì base64
- *Structured Extraction* — Trích xuất thông tin ra JSON từ ảnh/video
- *Visual Reasoning* — System prompt yêu cầu suy luận logic, không chỉ nhận diện

### Structured Output — JSON Schema

**RAG Response:**
```json
{
  "answer": "Câu trả lời tiếng Việt",
  "source": "Trang X, tài liệu Y",
  "confidence": "high | medium | low",
  "related_topics": ["topic1", "topic2"]
}
```

**Image Analysis Response:**
```json
{
  "image_type": "loại tài liệu",
  "extracted_data": {},
  "reasoning": "suy luận logic",
  "answer": "câu trả lời",
  "recommendations": ["gợi ý 1"],
  "confidence": "high | medium | low"
}
```

### System Prompt Design

Prompt được thiết kế với các nguyên tắc:

1. **Vai trò rõ ràng** — `"trợ lý sinh viên SGU"` → tránh lạc đề
2. **Ràng buộc context** — `"CHỈ dùng thông tin trong [CONTEXT]"` → chống hallucination
3. **JSON schema cứng** — Ví dụ output cụ thể → LLM ít lỗi format hơn
4. **Citation bắt buộc** — `"source"` phải có trang → có thể kiểm tra lại
5. **Confidence level** — Người dùng biết độ tin cậy của câu trả lời

### Token Management

| Chiến lược | Mô tả |
|-----------|-------|
| Model nhẹ | Gemini 2.0 Flash (không dùng Pro) |
| top_k=4 | Chỉ lấy 4 chunk context (~4000 ký tự) |
| max_tokens=1024 | Giới hạn output của LLM |
| Cache embedding | ChromaDB lưu disk, không embed lại |
| Cache component | `st.cache_resource` tránh init lại mỗi click |
| Fallback chain | Gemini → Groq → Ollama khi hết quota |

---

## 🛠️ Các Tool của Agent

| Tool | Input | Output |
|------|-------|--------|
| `search_document` | Query string | Chunks từ ChromaDB |
| `calculate_gpa` | `"Môn:điểm:TC,..."` | GPA hệ 4 + xếp loại |
| `get_current_date` | (rỗng) | Ngày hiện tại |
| `check_scholarship` | `"gpa:X,credits:N,..."` | Kết quả xét học bổng |

---

## 📊 Quota Miễn Phí

| Provider | Quota/ngày | Dùng cho |
|---------|-----------|---------|
| Gemini 2.0 Flash | 1.500 req | LLM chính |
| Gemini Embedding | 100 req/phút | Embedding (chỉ khi ingest) |
| Groq llama-3.1-8b | 14.400 req | LLM backup 1 |
| Ollama local | Không giới hạn | LLM backup 2 |
| ChromaDB | Không giới hạn | Vector DB |

---

## 🔧 Xử Lý Lỗi Thường Gặp

**Lỗi: "ChromaDB chưa được tạo"**
```bash
python src/ingest.py
```

**Lỗi: "Quota Gemini hết" (429)**

Hệ thống tự chuyển sang Groq → Ollama. Quota Gemini reset lúc 07:00 VN (00:00 UTC).

**Lỗi: "Ollama không kết nối được"**
```cmd
# Kiểm tra Ollama đang chạy
ollama list

# Khởi động thủ công (giữ cửa sổ CMD mở)
ollama serve

# Kiểm tra model đã pull chưa
ollama list
# Nếu chưa có: ollama pull qwen2.5:3b
```

**Lỗi: `'ollama' is not recognized`**

Restart máy tính sau khi cài Ollama, hoặc thêm thủ công vào PATH:
`C:\Users\<tên_user>\AppData\Local\Programs\Ollama`

**Lỗi: Import module**
```bash
# Đảm bảo chạy từ thư mục gốc
cd rag-chatbot-sgu
streamlit run app.py
```

---

## 📚 Tài Liệu Tham Khảo Cookbook

| Kỹ thuật | Notebook nguồn |
|---------|---------------|
| File API (Video/Audio) | `gemini-api/cookbook/video_understanding.ipynb` |
| JSON Mode | `gemini-api/cookbook/json_mode.ipynb` |
| Embeddings & RAG | `gemini-api/cookbook/embeddings/` |
| System Instructions | `gemini-api/cookbook/system_instructions.ipynb` |

Nguồn: https://github.com/google-gemini/cookbook

---

*Đồ án sử dụng dữ liệu: Sổ tay Hỗ trợ Sinh viên SGU 2022 + Học phí 2025–2026*