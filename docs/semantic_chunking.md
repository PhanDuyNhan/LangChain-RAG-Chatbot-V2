# Semantic Chunking — Tài liệu giải thích module

> **File code:** [`src/semantic_chunking.py`](../src/semantic_chunking.py)
****
---

## 1. Mục tiêu

Module này demo kỹ thuật **Semantic Chunking** trong LangChain — tách văn bản thành các đoạn (chunks) dựa trên **ngữ nghĩa** thay vì cắt cứng theo số ký tự.

Cụ thể, chúng ta:
1. Dùng `SemanticChunker` (từ `langchain_experimental.text_splitter`).
2. Áp dụng lên 1 đoạn văn bản mẫu chứa **2 chủ đề khác nhau** (LangChain/RAG và thị trường chứng khoán).
3. So sánh **2 cấu hình `breakpoint_threshold_amount`** — `95` (ngưỡng cao) vs `60` (ngưỡng thấp).
4. Quan sát số lượng chunks tạo ra ở mỗi cấu hình → **hiểu rõ ảnh hưởng của tham số ngưỡng**.

---

## 2. Cách chạy

```bash
.venv/bin/python src/semantic_chunking.py
```

Yêu cầu:
- Đã cài `langchain-experimental==0.3.4` (đã thêm trong [`requirements.txt`](../requirements.txt)).
- Có biến `GOOGLE_API_KEY` trong file `.env` (dùng chung với app chính).

---

## 3. Kiến trúc module

Module được thiết kế đơn giản, **độc lập hoàn toàn** với phần code RAG chatbot chính, gồm 4 thành phần:

| Thành phần | Vai trò |
|---|---|
| `SAMPLE_TEXT` | Hằng số chứa đoạn văn bản mẫu (copy nguyên từ đề bài). |
| `build_embeddings()` | Khởi tạo `GoogleGenerativeAIEmbeddings` với model `gemini-embedding-001`. |
| `run_config(embeddings, threshold, label)` | Chạy `SemanticChunker` với **1 ngưỡng**, in kết quả, trả về số chunks. |
| `main()` | Orchestrator — gọi 2 lần `run_config` (95, 60) rồi in phần giải thích. |

Sơ đồ luồng xử lý:

```
SAMPLE_TEXT
    │
    ▼
build_embeddings()  ──►  GoogleGenerativeAIEmbeddings
    │
    ▼
run_config(threshold=95)  ──►  SemanticChunker  ──►  N₁ chunks
run_config(threshold=60)  ──►  SemanticChunker  ──►  N₂ chunks
    │
    ▼
In bảng kết quả + Giải thích
```

---

## 4. Thuật toán Semantic Chunking — Cách hoạt động

`SemanticChunker` là **kỹ thuật nâng cao** so với `RecursiveCharacterTextSplitter` truyền thống. Thay vì cắt theo số ký tự cố định, nó:

### Bước 1 — Tách câu
Tách văn bản đầu vào thành danh sách câu đơn lẻ (split theo `.`, `?`, `!`).

### Bước 2 — Embedding từng câu
Gọi embedding model (Gemini ở đây) để chuyển mỗi câu thành **vector ngữ nghĩa** (768 chiều với `gemini-embedding-001`).

### Bước 3 — Tính khoảng cách ngữ nghĩa giữa các câu liền kề
Với mỗi cặp câu liền kề `(câu_i, câu_{i+1})`, tính **cosine distance** giữa 2 vector:

```
distance(i) = 1 - cosine_similarity(embed(câu_i), embed(câu_{i+1}))
```

→ Khoảng cách **càng lớn** = 2 câu **càng khác chủ đề**.

### Bước 4 — Xác định ngưỡng ngắt (breakpoint threshold)
Đây là bước **quyết định** số chunks, được điều khiển bởi 2 tham số:

| Tham số | Giá trị dùng | Ý nghĩa |
|---|---|---|
| `breakpoint_threshold_type` | `"percentile"` | Lấy ngưỡng theo **phân vị** của tập khoảng cách. |
| `breakpoint_threshold_amount` | `95` hoặc `60` | Số phân vị (0–100). Càng cao = ngưỡng càng cao = càng ít điểm vượt ngưỡng. |

Ví dụ với `breakpoint_threshold_amount=95`:
- Tính `threshold = percentile(distances, 95)` → ngưỡng = giá trị tại phân vị 95.
- Chỉ những điểm có `distance > threshold` (top 5% lớn nhất) mới được chọn làm **điểm ngắt**.

Với `breakpoint_threshold_amount=60`:
- `threshold = percentile(distances, 60)` → ngưỡng thấp hơn nhiều.
- Top 40% điểm có khoảng cách lớn nhất đều được chọn làm điểm ngắt → tạo ra **nhiều chunks hơn**.

### Bước 5 — Gộp các câu thành chunks
Đi qua danh sách câu, mỗi khi gặp 1 điểm ngắt thì kết thúc chunk hiện tại và mở chunk mới.

---

## 5. Dữ liệu đầu vào

Đoạn văn bản mẫu (theo đề bài) chứa **2 chủ đề rõ rệt**:

| Câu | Nội dung tóm tắt | Chủ đề |
|---|---|---|
| 1 | LangChain là framework mã nguồn mở cho LLM | LangChain |
| 2 | Cung cấp module quản lý prompt, kết nối LLM | LangChain |
| 3 | Một ứng dụng phổ biến là RAG | RAG |
| 4 | RAG kết hợp LLM với vector DB | RAG |
| 5 | Trong một diễn biến khác, thị trường chứng khoán biến động | Chứng khoán |
| 6 | VN-Index giảm nhẹ cuối phiên giao dịch | Chứng khoán |
| 7 | Nhà đầu tư thận trọng trước thông tin vĩ mô | Chứng khoán |

→ **Điểm chuyển chủ đề rõ nhất** nằm giữa câu 4 và câu 5 (LangChain/RAG → Chứng khoán).

---

## 6. Kết quả thực tế

| Cấu hình | `breakpoint_threshold_amount` | Số chunks | Đề dự kiến |
|:---:|:---:|:---:|:---:|
| 1 (ngưỡng cao) | **95** | **2** | 2 |
| 2 (ngưỡng thấp) | **60** | **3** | 3 hoặc 4 |

### Chi tiết các chunks tạo ra

**Cấu hình 1 — `threshold=95` (2 chunks):**
- **Chunk 1**: Câu 1 → Câu 4 *(toàn bộ phần LangChain + RAG)*
- **Chunk 2**: Câu 5 → Câu 7 *(toàn bộ phần Chứng khoán)*

**Cấu hình 2 — `threshold=60` (3 chunks):**
- **Chunk 1**: Câu 1 → Câu 2 *(giới thiệu LangChain)*
- **Chunk 2**: Câu 3 → Câu 5 *(RAG + câu chuyển)*
- **Chunk 3**: Câu 6 → Câu 7 *(Chứng khoán)*

---

## 7. Giải thích — Tại sao số chunks khác nhau?

### Khi `breakpoint_threshold_amount = 95`:
- Ngưỡng cao = **chỉ ngắt khi sự khác biệt ngữ nghĩa cực lớn** (top 5%).
- Trong đoạn văn bản mẫu, **chỉ có 1 vị trí thỏa mãn**: chỗ chuyển từ chủ đề "LangChain/RAG" sang "Chứng khoán" (giữa câu 4 và 5).
- → Tạo ra **2 chunks**, mỗi chunk **mạch lạc về 1 chủ đề duy nhất**.
- ✅ Phù hợp khi cần chunks **dài, có ngữ cảnh đầy đủ** (vd: tóm tắt tài liệu).

### Khi `breakpoint_threshold_amount = 60`:
- Ngưỡng thấp = **nhạy hơn nhiều** — top 40% khoảng cách lớn nhất đều thành điểm ngắt.
- Splitter phát hiện cả những "chuyển ý nhỏ" trong cùng 1 chủ đề (vd: từ "giới thiệu LangChain" → "ứng dụng RAG cụ thể").
- → Tạo ra **3 chunks ngắn hơn**.
- ✅ Phù hợp khi cần chunks **nhỏ, chi tiết** (vd: RAG retrieval với top-k cao).

### Ý nghĩa thực tiễn
**Trade-off** giữa 2 cấu hình:
- **Ngưỡng cao** → Ít chunks, ngữ cảnh đầy đủ, **nhưng có thể quá dài** với context window của LLM.
- **Ngưỡng thấp** → Nhiều chunks ngắn, dễ retrieval chính xác, **nhưng dễ mất ngữ cảnh** giữa các chunks.

→ Trong dự án thực tế, cần **tinh chỉnh `breakpoint_threshold_amount`** theo:
- Loại tài liệu (cẩm nang sinh viên, văn bản pháp luật, tin tức…).
- Kích thước context window của LLM dùng để generate.
- Yêu cầu chính xác của truy vấn người dùng.

---

## 8. So sánh với phương pháp truyền thống trong dự án

Phần ingest chính của project ([`src/ingest.py`](../src/ingest.py)) đang dùng `RecursiveCharacterTextSplitter` (cắt theo ký tự, `chunk_size=1000, chunk_overlap=200`). So sánh:

| Tiêu chí | `RecursiveCharacterTextSplitter` | `SemanticChunker` |
|---|---|---|
| **Cách cắt** | Theo số ký tự cố định | Theo ngữ nghĩa (cosine distance giữa câu) |
| **Chi phí** | Rất rẻ (chỉ split chuỗi) | Tốn API embedding cho từng câu |
| **Tốc độ** | Rất nhanh | Chậm hơn nhiều (gọi API) |
| **Chất lượng chunks** | Có thể cắt giữa câu/đoạn | Chunks luôn mạch lạc về chủ đề |
| **Phù hợp khi** | Dữ liệu lớn, cần tốc độ | Dữ liệu ngắn, cần chất lượng cao |

→ `SemanticChunker` là **kỹ thuật nâng cao**, đánh đổi tốc độ + chi phí lấy chất lượng chunks tốt hơn.

---

## 9. Phụ thuộc & môi trường

| Thư viện | Version | Mục đích |
|---|---|---|
| `langchain-experimental` | `0.3.4` | Cung cấp class `SemanticChunker` |
| `langchain-google-genai` | `2.1.4` | Wrapper Gemini Embedding |
| `langchain-core` | `0.3.83` | Core types |
| `python-dotenv` | `1.1.0` | Load `.env` |

Model embedding dùng: `models/gemini-embedding-001` *(model duy nhất hỗ trợ `embedContent` với API key của project)*.

---

## 10. Tham khảo

- LangChain docs — [Semantic Chunking](https://python.langchain.com/docs/how_to/semantic-chunker/)
- Greg Kamradt — [5 Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) *(nguồn cảm hứng của `SemanticChunker`)*
