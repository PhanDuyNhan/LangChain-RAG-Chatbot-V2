<<<<<<< HEAD
<<<<<<< HEAD
# 🎓 SGU Chatbot - RAG Hỗ trợ Sinh viên
**Đồ án môn: Các Công nghệ Lập trình Hiện đại**  
**Stack: LangChain + Gemini 2.0 Flash + ChromaDB + Streamlit**  
**Chi phí: MIỄN PHÍ HOÀN TOÀN**

---

## 📁 Cấu Trúc Dự Án

```
rag-chatbot-sgu/
├── data/                          # Đặt file PDF/DOCX tại đây
│   └── cam_nang_sinh_vien.pdf    # File đã có sẵn
├── src/
│   ├── ingest.py         # [LEVEL 1] Load → Chunk → Embed → ChromaDB
│   ├── llm_router.py     # Token Management: Gemini → Groq → Ollama
│   ├── rag_chain.py      # [LEVEL 1] RAG Pipeline + Structured JSON Output
│   ├── multimodal.py     # [LEVEL 2] Phân tích ảnh với Gemini Vision
│   └── agent.py          # [LEVEL 3] ReAct Agent + 4 custom tools
├── app.py                # Streamlit UI (3 tab)
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

**Gemini API Key** (bắt buộc):
1. Truy cập: https://aistudio.google.com/app/apikey
2. Click "Create API Key" → Copy key
3. Dán vào `.env`: `GOOGLE_API_KEY=AIza...`

**Groq API Key** (backup, khuyến nghị):
1. Truy cập: https://console.groq.com/keys
2. Tạo key miễn phí → Copy
3. Dán vào `.env`: `GROQ_API_KEY=gsk_...`

### Bước 3 (tùy chọn): Cài Ollama trên Windows (backup local, không giới hạn)

> Ollama là LLM chạy hoàn toàn trên máy tính của bạn — không cần internet, không giới hạn request.  
> Dùng làm backup khi Gemini và Groq hết quota trong ngày.

**3.1 — Tải và cài đặt Ollama:**
1. Truy cập: **https://ollama.com/download**
2. Click **"Download for Windows"** → tải file `OllamaSetup.exe` (~50MB)
3. Chạy file `.exe` → Next → Install → Finish
4. Ollama tự chạy nền sau khi cài (icon xuất hiện ở system tray góc phải taskbar)

> ⚠️ Nếu Windows Defender chặn: Click **"More info"** → **"Run anyway"**

**3.2 — Kiểm tra cài đặt thành công:**

Mở **Command Prompt** (Win + R → gõ `cmd` → Enter):
```cmd
ollama --version
```
Hiện ra `ollama version 0.x.x` là thành công ✅  
Nếu báo `'ollama' is not recognized`: **restart máy** rồi thử lại.

**3.3 — Download model qwen2.5:3b:**
```cmd
ollama pull qwen2.5:3b
```
> ⏳ Chờ **5–15 phút** (model ~1.9GB, tùy tốc độ mạng)  
> Sẽ hiện progress bar: `pulling manifest... 45%... 100%`  
> Khi xong hiện chữ `success` là hoàn tất ✅

**3.4 — Test model hoạt động:**
```cmd
ollama run qwen2.5:3b
```
Gõ thử `Xin chào` → model trả lời được là OK ✅  
Thoát bằng lệnh: `/bye`

**3.5 — Giữ Ollama chạy nền khi dùng chatbot:**

Ollama cần đang chạy khi app gọi đến. Có 2 cách:
- **Tự động**: Sau khi cài, Ollama tự khởi động cùng Windows → kiểm tra icon ở system tray
- **Thủ công**: Mở CMD và chạy lệnh sau, **giữ cửa sổ này mở**:
```cmd
ollama serve
```

**3.6 — Kiểm tra Ollama hoạt động với Python:**

Mở CMD tại thư mục `rag-chatbot-sgu/`, chạy:
```cmd
python -c "from langchain_ollama import ChatOllama; llm = ChatOllama(model='qwen2.5:3b'); print(llm.invoke('Xin chào').content)"
```
In ra câu trả lời tiếng Việt → **hoàn tất** ✅

**Yêu cầu phần cứng tối thiểu cho qwen2.5:3b:**
| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|-------------|
| RAM | 8GB | 16GB |
| Dung lượng disk | 3GB trống | 5GB trống |
| CPU | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| GPU | Không bắt buộc | NVIDIA (chạy nhanh hơn) |

### Bước 4: Nạp tài liệu vào ChromaDB
```bash
python src/ingest.py
```
*Lần đầu chạy sẽ gọi Gemini API để embedding (~30-60 giây)*

### Bước 5: Chạy ứng dụng
```bash
streamlit run app.py
```
*Mở trình duyệt tại: http://localhost:8501*

---

## 🏗️ Kiến Trúc Kỹ Thuật

### RAG Pipeline (Level 1)
```
PDF → PyPDFLoader → RecursiveTextSplitter(1000/200)
    → GoogleGenerativeAIEmbeddings(text-embedding-004)
    → ChromaDB (local persist)
    
Query → Embed → similarity_search(k=4) → Context
      → System Prompt + Context + Question → LLM
      → Parse JSON → {"answer","source","confidence","related_topics"}
```

### Token Management
| Chiến lược | Mô tả |
|-----------|-------|
| Model nhẹ | Gemini 2.0 Flash (không phải Pro) |
| top_k=4 | Chỉ lấy 4 chunk context (~4000 ký tự) |
| max_tokens=1024 | Giới hạn output của LLM |
| Cache embedding | ChromaDB lưu disk, không embed lại |
| Cache component | `st.cache_resource` tránh init lại |
| Fallback chain | Gemini → Groq → Ollama khi hết quota |

### Structured Output - JSON Schema
```json
{
  "answer": "Câu trả lời tiếng Việt",
  "source": "Trang X, tài liệu Y",
  "confidence": "high|medium|low",
  "related_topics": ["topic1", "topic2"]
}
```

### System Prompt Design
Prompt được thiết kế với:
1. **Vai trò rõ ràng**: "trợ lý sinh viên SGU" → tránh lạc đề
2. **Ràng buộc context**: "CHỈ dùng thông tin trong [CONTEXT]" → chống hallucination
3. **JSON schema cứng**: Ví dụ output cụ thể → LLM ít lỗi format hơn
4. **Citation bắt buộc**: "source" phải có trang → verifiable
5. **Confidence level**: Người dùng biết độ tin cậy

---

## 🛠️ Các Tool của Agent (Level 3)

| Tool | Input | Output |
|------|-------|--------|
| `search_document` | Query string | Chunks từ ChromaDB |
| `calculate_gpa` | "Môn:điểm:TC,..." | GPA hệ 4 + xếp loại |
| `get_current_date` | (rỗng) | Ngày hiện tại |
| `check_scholarship` | "gpa:X,credits:N,..." | Kết quả xét học bổng |

---

## 🔧 Xử Lý Lỗi Thường Gặp

**Lỗi: "ChromaDB chưa được tạo"**
```bash
python src/ingest.py
```

**Lỗi: "Quota Gemini hết"**
- Hệ thống tự chuyển sang Groq hoặc Ollama
- Quota reset lúc 07:00 VN (00:00 UTC)

**Lỗi: "Ollama không kết nối được" / "Connection refused"**
```cmd
:: Bước 1: Kiểm tra Ollama có đang chạy không
ollama list

:: Bước 2: Nếu không phản hồi, khởi động thủ công (giữ cửa sổ CMD mở)
ollama serve

:: Bước 3: Kiểm tra model đã pull chưa — phải thấy "qwen2.5:3b"
ollama list
:: Nếu chưa có: ollama pull qwen2.5:3b
```

**Lỗi: `'ollama' is not recognized as an internal or external command`**
- Restart máy tính sau khi cài Ollama
- Hoặc thêm thủ công vào PATH: `C:\Users\<tên_user>\AppData\Local\Programs\Ollama`

**Lỗi: Import module**
```bash
# Đảm bảo chạy từ thư mục gốc rag-chatbot-sgu/
cd rag-chatbot-sgu
streamlit run app.py
```

---

## 📊 Quota Miễn Phí

| Provider | Quota/ngày | Dùng cho |
|---------|-----------|---------|
| Gemini 2.0 Flash | 1,500 req | LLM chính |
| Gemini Embedding | 100 req/phút | Embedding (chỉ khi ingest) |
| Groq llama-3.1-8b | 14,400 req | LLM backup 1 |
| Ollama local | Không giới hạn | LLM backup 2 |
| ChromaDB | Không giới hạn | Vector DB |

---

*Đồ án sử dụng dữ liệu: Sổ tay Hỗ trợ Sinh viên SGU 2022 + Học phí 2025-2026*
=======
# LangChain-RAG-Chatbot-V2
>>>>>>> dd8cd39cc3cc280d931b27ab07e768715e17b258
=======

>>>>>>> 50481ee79c6c90fcca5815e908196f079221c791
