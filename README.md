# SGU Chatbot - RAG & Multimodal Ho Tro Sinh Vien

Do an mon `Cac Cong nghe Lap trinh Hien dai`.

Project tap trung vao 2 muc:
- `Muc 1`: RAG chatbot tra loi dua tren tai lieu rieng cua SGU
- `Muc 2`: Multimodal voi anh, video, audio bang Gemini

Stack chinh:
- `Streamlit`
- `LangChain`
- `Gemini API`
- `ChromaDB`
- `Groq` va `Ollama` dung lam fallback cho text chat

## 1. Tinh nang chinh

### Muc 1 - RAG chatbot
- Doc file PDF/DOCX, lam sach va chia chunk
- Tao embedding bang `gemini-embedding-001`
- Luu vector vao `ChromaDB`
- Truy xuat `top_k=4` chunk lien quan nhat
- Tra loi bang tieng Viet dua tren context da truy xuat
- Hien thi citation theo `trang + ten tai lieu`
- Chuan hoa structured output de UI co the hien thi va xu ly tiep

### Muc 2 - Multimodal
- Phan tich anh tai lieu SGU bang `Visual Reasoning`
- Trich xuat thong tin co cau truc tu anh
- Phan tich video/audio bang `Gemini File API`
- Upload media len server Google, cho xu ly, goi model qua `URI`
- Tra ket qua dang JSON de hien thi len web

### Toi uu quota Gemini free
- Rate limiter theo RPM
- Soft cap theo ngay de tranh het quota
- Retry voi exponential backoff khi gap `429`
- Cache response theo `question` hoac `file hash`
- Fallback text model: `Gemini -> Groq -> Ollama`

## 2. Cau truc project

```text
LangChain-RAG-Chatbot-V2/
├── app.py
├── quota_guard.py
├── requirements.txt
├── config.toml
├── README.md
├── data/
│   └── cam_nang_sinh_vien_v2.pdf
└── src/
    ├── ingest.py
    ├── llm_router.py
    ├── rag_chain.py
    ├── multimodal.py
    └── __init__.py
```

Y nghia file:
- `app.py`: giao dien Streamlit
- `quota_guard.py`: limiter, cache, soft cap, counter
- `src/ingest.py`: nap tai lieu, chunking, embedding, tao ChromaDB
- `src/rag_chain.py`: pipeline RAG cho Muc 1
- `src/llm_router.py`: router cho Gemini, Groq, Ollama
- `src/multimodal.py`: anh, video, audio cho Muc 2

## 3. Cai dat

### Buoc 1: Cai thu vien

```bash
pip install -r requirements.txt
```

### Buoc 2: Tao file `.env`

```env
# --- LLM CHÍNH: Google Gemini (miễn phí 1500 req/ngày) ---
GOOGLE_API_KEY=

# --- LLM BACKUP 1: Groq (miễn phí 14400 req/ngày) ---
GROQ_API_KEY=

# --- LLM BACKUP 2: Ollama (local, không giới hạn) ---
# Mặc định chạy tại http://localhost:11434
# Không cần key, chỉ cần cài Ollama và pull model:
#   ollama pull qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434

# --- CẤU HÌNH CHROMA DB (vector database local) ---
CHROMA_DB_PATH=./chroma_db

# --- CẤU HÌNH RAG ---
# Số chunk tối đa trả về khi tìm kiếm (top_k) - Token Management
RETRIEVER_TOP_K=4

# --- THƯ MỤC DỮ LIỆU ---
DATA_DIR=./data


Ghi chu:
- `GOOGLE_API_KEY`: bat buoc
- `GROQ_API_KEY`: khuyen nghi de fallback
- `OLLAMA_BASE_URL`: tuy chon
- `GEMINI_TEXT_MODEL`: project hien dang toi uu cho quota free `gemini-2.5-flash`

### Buoc 3: Nap du lieu vao ChromaDB

```bash
python src/ingest.py
```

Lan dau se:
- doc file trong thu muc `data/`
- chia chunk
- goi embedding API
- luu ket qua vao `chroma_db/`

### Buoc 4: Chay ung dung

```bash
streamlit run app.py
```

Mac dinh mo tai:
- `http://localhost:8501`

## 4. Kien truc ky thuat

### Muc 1 - RAG pipeline

```text
PDF/DOCX
-> Load document
-> Chunking
-> Gemini Embedding
-> ChromaDB

Question
-> Retrieve top_k chunk
-> Inject context vao prompt
-> Gemini / Groq / Ollama
-> Chuan hoa response
-> Hien thi answer + source + confidence + related_topics
```

Diem can nhan manh khi bao cao:
- co `chunking strategy`
- co `vector search`
- co `citation`
- co `structured output` de UI xu ly tiep

### Muc 2 - Anh

```text
Image
-> Resize toi da 1024px
-> Gemini multimodal
-> Visual reasoning
-> JSON:
   image_type
   extracted_data
   reasoning
   answer
   recommendations
   confidence
```

### Muc 2 - Video/Audio

```text
Media file
-> Upload bang Gemini File API
-> Cho file ACTIVE
-> Goi model qua file URI
-> JSON:
   content_type
   summary
   action_items
   key_moments
   answer
   confidence
-> Xoa file tren server sau khi xu ly
```

## 5. Structured output

### RAG response schema

```json
{
  "answer": "Noi dung tra loi",
  "source": "Trang 11, 13 (cam_nang_sinh_vien_v2.pdf)",
  "confidence": "high",
  "related_topics": [
    "Thu tuc xin hoan thi?",
    "Dieu kien va quy trinh hoan thi"
  ],
  "provider": "Gemini gemini-2.5-flash",
  "from_cache": false,
  "output_format": "json_schema_normalized",
  "schema_version": "rag_v1"
}
```

### Image response schema

```json
{
  "image_type": "LICH THI",
  "extracted_data": {},
  "reasoning": "Mon thi som nhat ...",
  "answer": "Ban thi mon ...",
  "recommendations": [
    "On tap som",
    "Kiem tra phong thi"
  ],
  "confidence": "high"
}
```

### Media response schema

```json
{
  "content_type": "meeting_audio",
  "summary": [
    "Diem 1",
    "Diem 2"
  ],
  "action_items": [
    {
      "task": "Viec can lam",
      "assignee": "Nguoi phu trach",
      "deadline": "Han chot"
    }
  ],
  "key_moments": [
    "00:35",
    "03:10"
  ],
  "answer": "Tom tat ngan gon",
  "confidence": "high"
}
```

## 6. Quota va chi phi

Project duoc toi uu cho Gemini free tier.

Thiet lap dang dung trong app:
- Text model: `gemini-2.5-flash`
- Soft cap text: `4 RPM / 18 req-ngay`
- Embedding soft cap: `60 RPM / 900 req-ngay`
- Vision/File API soft cap: `3 RPM / 18 req-ngay`

Ly do khong de sat gioi han:
- tranh `429`
- tranh burst request tu Streamlit
- uu tien demo on dinh hon la tan dung toi da quota

Neu Gemini vuot gioi han:
- Muc 1 se fallback sang `Groq`
- Neu can co the fallback tiep sang `Ollama`

## 7. Cach demo

### Demo Muc 1
1. Dat cau hoi ve hoc vu, hoc phi, hoan thi, tot nghiep
2. Cho thay answer
3. Mo `Structured Output JSON`
4. Mo `Context da tim thay`
5. Nhan manh citation theo trang/tai lieu

Vi du:
- `Thu tuc xin hoan thi?`
- `Dieu kien tot nghiep?`
- `Hoc phi CNTT nam 2025?`

### Demo Muc 2
1. Upload anh lich thi / thong bao / hoa don
2. Dat cau hoi yeu cau suy luan
3. Cho thay:
   - answer
   - reasoning
   - extracted_data
   - JSON response

Vi du:
- `Lich thi nay toi can chuan bi gi? Mon nao thi som nhat?`
- `Thong bao nay yeu cau toi lam gi va deadline la khi nao?`

### Demo video/audio
1. Upload file audio/video ngan
2. Dat cau hoi tom tat noi dung
3. Cho thay:
   - summary
   - action_items
   - key_moments

## 8. Lien he voi rubric do an

### Muc 1
- Data pipeline: co
- Chunking strategy: co
- Embedding + vector DB: co
- Context injection: co
- Citation: co

### Muc 2
- Visual reasoning: co
- Gemini File API: co
- Structured extraction JSON: co

Project hien tap trung hoan thien chac `Muc 1 + Muc 2`.
Chua trien khai `Muc 3` va `Muc 4`.

## 9. Loi thuong gap

### Chua co ChromaDB

```bash
python src/ingest.py
```

### Het quota Gemini / loi 429
- App da co limiter va backoff
- Thu lai sau
- Neu can, Muc 1 se fallback sang Groq/Ollama

### Sidebar hien text ky thuat la
- Day la loi render Streamlit da duoc sua trong `app.py`
- Chi can restart app neu ban dang chay ban cu

### Ollama khong chay

```bash
ollama serve
ollama list
```

## 10. Thu vien su dung

Tu `requirements.txt`:
- `langchain`
- `langchain-community`
- `langchain-google-genai`
- `langchain-groq`
- `langchain-ollama`
- `langchain-chroma`
- `google-genai`
- `chromadb`
- `streamlit`
- `pypdf`
- `docx2txt`
- `Pillow`
- `requests`
- `python-dotenv`

## 11. Ghi chu

- Du lieu demo hien tai: `cam_nang_sinh_vien_v2.pdf`
- README nay mo ta dung theo trang thai code hien tai cua repo
- Neu bao cao do an, nen nhan manh:
  - RAG
  - Citation
  - Visual Reasoning
  - Gemini File API
  - Structured Output
  - Token Management / Cost Optimization
