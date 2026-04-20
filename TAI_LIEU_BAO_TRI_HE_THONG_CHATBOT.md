# Tài liệu bảo trì hệ thống Chatbot

Tài liệu này dùng như một "bản đồ hệ thống" để sau này sửa lỗi, mở rộng tính năng, hoặc đổi model mà biết phải tìm đúng chỗ nào trước. Phần được giải thích kỹ nhất là luồng `LangChain + RAG`, vì đây là lõi của chatbot tài liệu.

## 1. Mục tiêu của hệ thống

Project này hiện có 2 phần chính:

1. `Mức 1 - Chat tài liệu (RAG)`: trả lời câu hỏi dựa trên tài liệu SGU đã nạp vào `ChromaDB`.
2. `Mức 2 - Multimodal`: phân tích ảnh, video, audio bằng Gemini.

Nếu chỉ xét chatbot hỏi đáp tài liệu, lõi chạy nằm ở:

- `app.py`
- `src/ingest.py`
- `src/rag_chain.py`
- `src/llm_router.py`
- `quota_guard.py`

## 2. Sơ đồ tổng quan

```text
Người dùng nhập câu hỏi trên Streamlit
-> app.py
-> RAGChain.query()
-> kiểm tra cache
-> query embedding
-> tìm top_k chunks trong ChromaDB
-> đánh giá độ liên quan
-> nếu đủ liên quan: tạo prompt có context
-> gọi LLM (Gemini -> Groq -> Ollama)
-> parse JSON trả về
-> render answer + source + confidence + related_topics trên UI
```

Luồng nạp dữ liệu ban đầu:

```text
File PDF/DOCX trong data/
-> src/ingest.py: load_documents()
-> split_documents()
-> GeminiEmbeddings
-> Chroma.from_documents(...)
-> lưu vào chroma_db/
```

## 3. Cấu trúc file quan trọng

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit, nhận input, gọi RAG, hiển thị kết quả |
| `src/ingest.py` | Đọc tài liệu, chunking, embedding, tạo và load ChromaDB |
| `src/rag_chain.py` | Lõi RAG: retrieve, kiểm tra relevance, fallback, gọi LLM, parse output |
| `src/llm_router.py` | Khởi tạo và fallback giữa Gemini, Groq, Ollama |
| `quota_guard.py` | Rate limiter, retry 429, cache, counter, soft cap theo ngày |
| `src/multimodal.py` | Logic cho ảnh, video, audio |
| `data/` | Nơi đặt tài liệu nguồn để index |
| `chroma_db/` | Vector DB local của Chroma |
| `.quota_usage.json` | File lưu usage theo ngày để không đốt quota sau khi restart app |

Tại thời điểm mình đọc code, dữ liệu đang có:

- `data/cam_nang_sinh_vien_v2.pdf`
- `chroma_db/chroma.sqlite3`

## 4. Điểm bắt đầu khi đọc hệ thống

Nếu muốn hiểu nhanh project, đọc theo thứ tự này:

1. `app.py`
2. `src/rag_chain.py`
3. `src/ingest.py`
4. `src/llm_router.py`
5. `quota_guard.py`

Lý do:

- `app.py` cho biết người dùng thao tác thế nào.
- `src/rag_chain.py` cho biết chatbot quyết định trả lời ra sao.
- `src/ingest.py` cho biết dữ liệu được biến thành vector thế nào.
- `src/llm_router.py` và `quota_guard.py` cho biết vì sao model đổi provider hoặc bị chặn quota.

## 5. Phần LangChain và RAG

### 5.1. Hệ thống đang dùng LangChain ở đâu

Project này có dùng LangChain, nhưng không theo kiểu ghép `RetrievalQA` rất ngắn gọn. Thay vào đó, tác giả tự dựng pipeline để kiểm soát quota, fallback và JSON output.

Các thành phần LangChain chính:

- `langchain_community.document_loaders.PyPDFLoader` và `Docx2txtLoader` trong `src/ingest.py:27-28`
- `RecursiveCharacterTextSplitter` trong `src/ingest.py:28`
- `langchain_chroma.Chroma` trong `src/ingest.py:31` và `src/rag_chain.py:30`
- `langchain_core.documents.Document` trong `src/ingest.py` và `src/rag_chain.py`
- `langchain_core.messages.HumanMessage` trong `src/rag_chain.py:32`
- `ChatGoogleGenerativeAI`, `ChatGroq`, `ChatOllama` trong `src/llm_router.py`

Nói ngắn gọn:

- LangChain lo phần loader, splitter, vector store wrapper, chat model wrapper.
- Logic nghiệp vụ chính vẫn nằm trong code custom của project.

### 5.2. Luồng nạp dữ liệu RAG

File chính: `src/ingest.py`

Các hàm cần nhớ:

- `load_documents()` ở `src/ingest.py:74`
- `split_documents()` ở `src/ingest.py:121`
- `GeminiEmbeddings` ở `src/ingest.py:198`
- `create_embeddings()` ở `src/ingest.py:293`
- `build_vector_store()` ở `src/ingest.py:300`
- `load_vector_store()` ở `src/ingest.py:312`
- `check_db_exists()` ở `src/ingest.py:320`

Luồng xử lý:

1. `load_documents(DATA_DIR)` đọc tất cả file `.pdf` và `.docx` trong `data/`.
2. `split_documents(documents)` chia thành chunks.
3. `GeminiEmbeddings.embed_documents()` gọi Gemini Embedding REST API.
4. `Chroma.from_documents(...)` lưu toàn bộ vector vào `chroma_db/`.

Điểm quan trọng của chunking:

- Nếu file có pattern `KB001`, `KB002`... thì code cố giữ nguyên block kiến thức bằng regex.
- Nếu là PDF thường thì dùng `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.

Điều này có nghĩa là:

- Nếu tài liệu mới có format giống "knowledge base" thì nên giữ pattern `KB###`.
- Nếu retrieval kém, lỗi thường nằm ở `split_documents()` trước khi nằm ở model.

### 5.3. Luồng query RAG

File chính: `src/rag_chain.py`

Các điểm vào quan trọng:

- `RAGChain.__init__()` quanh `src/rag_chain.py:443`
- `_init_components()` ở `src/rag_chain.py:450`
- `_retrieve_docs_with_relevance()` ở `src/rag_chain.py:609`
- `_invoke_with_fallback()` ở `src/rag_chain.py:657`
- `_general_fallback_response()` ở `src/rag_chain.py:731`
- `_parse_json_response()` ở `src/rag_chain.py:788`
- `query()` ở `src/rag_chain.py:824`

Luồng `query()` thực tế:

1. Nhận câu hỏi từ UI.
2. Check `_rag_cache` trước.
3. Retrieve chunks từ `Chroma`.
4. Tính `best_relevance` và các score liên quan.
5. Quyết định:
   - Nếu câu hỏi không khớp tài liệu -> `general_fallback`
   - Nếu khớp -> chạy RAG bình thường
6. Tạo prompt có `[CONTEXT]`.
7. Gọi LLM qua `_invoke_with_fallback()`.
8. Parse JSON, sửa output nếu model trả sai format.
9. Trả object chuẩn cho UI.

### 5.4. Cách retrieve đang hoạt động

Retriever được khởi tạo ở `src/rag_chain.py:459-463`:

- `self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k})`

Thông số chính:

- `RETRIEVER_TOP_K` mặc định là `4`
- `RAG_MIN_RELEVANCE` mặc định là `0.32`

Logic chấm độ liên quan không chỉ dựa vào 1 nguồn:

1. `similarity_search_with_relevance_scores()`
2. Nếu không được thì `similarity_search_with_score()`
3. Nếu vẫn không được thì `retriever.invoke()`
4. Sau đó cộng thêm `keyword overlap` tự viết tay

Tức là độ liên quan cuối cùng là logic lai giữa:

- score của Chroma/LangChain
- overlap từ khóa do project tự tính

Đây là chỗ rất quan trọng nếu sau này chatbot "bốc nhầm tài liệu".

### 5.5. Cách hệ thống quyết định có dùng RAG hay không

Hàm chính: `_should_general_fallback()` ở `src/rag_chain.py:571`

Mục đích:

- Nếu câu hỏi là kiến thức chung, không phải SGU, hoặc retrieval quá yếu, thì không ép trả lời theo tài liệu.
- Khi đó hệ thống chuyển sang `GENERAL_FALLBACK_PROMPT`.

Các trường hợp dễ rơi vào fallback:

- Không có documents
- `best_relevance` thấp hơn ngưỡng
- Có từ khóa chung chung nhưng thiếu từ khóa đặc thù
- Câu hỏi mang tính kiến thức phổ thông, không có intent SGU

Đây là phần rất quan trọng để giảm hallucination.

### 5.6. Prompt RAG nằm ở đâu

Prompt chính nằm đầu file `src/rag_chain.py`:

- `SYSTEM_PROMPT`
- `GENERAL_FALLBACK_PROMPT`
- `JSON_RETRY_SUFFIX`

Muốn chỉnh hành vi trả lời thì thường sửa ở đây:

- đổi giọng văn
- đổi cách citation
- thêm trường JSON
- ép model trả lời ngắn hơn/dài hơn
- siết chặt chống bịa

### 5.7. Cách gọi model trong LangChain

File: `src/llm_router.py`

Các hàm quan trọng:

- `_try_gemini()` ở `src/llm_router.py:30`
- `_try_groq()` ở `src/llm_router.py:43`
- `_try_ollama()` ở `src/llm_router.py:55`
- `get_llm()` ở `src/llm_router.py:67`

Thứ tự fallback hiện tại:

1. `Gemini`
2. `Groq`
3. `Ollama`

Trong `src/rag_chain.py`, `_invoke_with_fallback()` sẽ:

- lấy LLM từ cache nếu đã khởi tạo trước đó
- gọi `.invoke([HumanMessage(content=prompt)])`
- retry khi gặp quota/429
- nếu JSON hỏng thì retry lại với `JSON_RETRY_SUFFIX`
- nếu provider chết thì chuyển provider tiếp theo

### 5.8. Output schema của RAG

Schema chuẩn mà UI đang mong đợi:

```json
{
  "answer": "...",
  "source": "...",
  "confidence": "high|medium|low",
  "related_topics": ["..."],
  "provider": "...",
  "from_cache": false,
  "response_mode": "rag|general_fallback",
  "relevance_score": 0.0,
  "output_format": "json_schema_normalized",
  "schema_version": "rag_v1"
}
```

Chỗ parse/sửa output nằm ở:

- `_extract_json_fragment()`
- `_normalize_rag_payload()`
- `_parse_json_response()`

Nếu sau này thêm field mới thì phải sửa đồng bộ ở:

1. prompt schema trong `SYSTEM_PROMPT`
2. `RAGResponseSchema`
3. `_normalize_rag_payload()`
4. `_parse_json_response()`
5. `_render_rag_response()` trong `app.py`

## 6. UI đang nối với RAG thế nào

File chính: `app.py`

Các hàm cần nhớ:

- `load_rag_chain()` ở `app.py:153`
- `render_sidebar()` ở `app.py:166`
- `render_tab_chat()` ở `app.py:287`
- `_render_rag_response()` ở `app.py:406`

Luồng UI:

1. Người dùng nhập câu hỏi ở `st.chat_input`.
2. `render_tab_chat()` gọi `load_rag_chain()`.
3. `load_rag_chain()` tạo `RAGChain()` và cache bằng `@st.cache_resource`.
4. Gọi `rag.query(question)`.
5. Kết quả được thêm vào `st.session_state.chat_history`.
6. `_render_rag_response()` hiển thị:
   - answer
   - source
   - confidence
   - provider
   - relevance_score
   - related_topics
   - structured JSON
   - retrieved chunks

Nếu sau này UI hiển thị sai dù backend đúng, xem `app.py` trước khi đụng `src/rag_chain.py`.

## 7. Quota, retry, cache nằm ở đâu

File chính: `quota_guard.py`

Thành phần quan trọng:

- `RateLimiter` ở `quota_guard.py:32`
- `with_retry()` ở `quota_guard.py:120`
- `SimpleCache` ở `quota_guard.py:158`
- `DailyQuotaTracker` ở `quota_guard.py:248`

Ứng dụng thực tế:

- `RAG cache`: cache câu hỏi text
- `Vision cache`: cache cho multimodal
- `DailyQuotaTracker`: lưu usage theo ngày ở `.quota_usage.json`
- `RateLimiter`: khống chế tốc độ gọi Gemini
- `with_retry`: retry các lỗi 429/quota

Nếu gặp lỗi kiểu:

- đang chạy bình thường rồi tự nhiên chậm
- bị chờ lâu giữa các request
- hôm nay app cứ báo soft cap

thì kiểm tra `quota_guard.py` và `.quota_usage.json` trước.

## 8. Biến môi trường quan trọng

Những biến đang được dùng nhiều:

| Biến | Ý nghĩa |
|---|---|
| `GOOGLE_API_KEY` | API key cho Gemini text + embedding + multimodal |
| `GROQ_API_KEY` | API key fallback text |
| `OLLAMA_BASE_URL` | URL của Ollama local |
| `CHROMA_DB_PATH` | thư mục lưu ChromaDB |
| `DATA_DIR` | thư mục chứa tài liệu nguồn |
| `RETRIEVER_TOP_K` | số chunks lấy ra khi search |
| `RAG_MIN_RELEVANCE` | ngưỡng để quyết định có fallback khỏi tài liệu hay không |
| `RAG_CACHE_NAMESPACE` | namespace cache RAG |
| `GEMINI_TEXT_MODEL` | model Gemini text đang dùng |
| `GEMINI_MAX_OUTPUT_TOKENS` | max tokens cho Gemini text |
| `GEMINI_LLM_RPM`, `GEMINI_LLM_RPD_SOFT` | quota text |
| `GEMINI_EMBED_RPM`, `GEMINI_EMBED_RPD_SOFT` | quota embedding |
| `GEMINI_VISION_RPM`, `GEMINI_VISION_RPD_SOFT` | quota vision/file API |
| `QUOTA_STATE_FILE` | file lưu usage theo ngày |

## 9. Khi cần sửa lỗi thì tìm ở đâu

### 9.1. Chatbot không tìm thấy dữ liệu / báo chưa có ChromaDB

Xem:

- `app.py:158` `check_db_ready()`
- `src/ingest.py:320` `check_db_exists()`
- thư mục `chroma_db/`

Thường xử lý bằng:

```bash
python src/ingest.py
```

### 9.2. Trả lời sai nguồn hoặc không có citation

Xem:

- `_format_context()` trong `src/rag_chain.py:482`
- `_build_source_from_docs()` trong `src/rag_chain.py`
- `_render_rag_response()` trong `app.py:406`

### 9.3. Tìm đúng tài liệu nhưng câu trả lời nghèo hoặc thiếu ý

Xem:

- `SYSTEM_PROMPT` trong `src/rag_chain.py`
- `RETRIEVER_TOP_K`
- `split_documents()` trong `src/ingest.py`

Nguyên nhân thường gặp:

- chunk quá nhỏ
- context bị cắt thiếu
- prompt chưa ép "liệt kê đủ ý"

### 9.4. Chatbot trả lời kiến thức chung thay vì bám tài liệu

Xem:

- `_should_general_fallback()` trong `src/rag_chain.py:571`
- `RAG_MIN_RELEVANCE`
- `_GENERAL_KNOWLEDGE_PATTERNS`
- `_SGU_INTENT_TERMS`

### 9.5. Model bị đổi từ Gemini sang Groq/Ollama

Xem:

- `src/llm_router.py`
- `_invoke_with_fallback()` trong `src/rag_chain.py:657`
- `quota_guard.py`

Nguyên nhân thường là:

- hết quota Gemini
- lỗi 429
- thiếu API key
- Ollama local không chạy

### 9.6. Model trả JSON lỗi, UI không render đúng

Xem:

- `_parse_json_response()` ở `src/rag_chain.py:788`
- `RAGResponseSchema`
- `JSON_RETRY_SUFFIX`
- `_render_rag_response()` ở `app.py:406`

### 9.7. Retrieval kém, tìm sai chunk

Xem:

- `split_documents()` ở `src/ingest.py:121`
- `_retrieve_docs_with_relevance()` ở `src/rag_chain.py:609`
- `RETRIEVER_TOP_K`
- `RAG_MIN_RELEVANCE`

Đây là lỗi rất hay gặp trong RAG, và thường nên sửa ở dữ liệu/chunking trước khi đổ cho model.

### 9.8. Embedding bị lỗi hoặc ingest bị 429

Xem:

- `GeminiEmbeddings` ở `src/ingest.py:198`
- `quota_guard.py`

Lưu ý:

- embedding đang gọi thẳng REST API, không dùng wrapper embed sẵn của LangChain
- có throttle, jitter, batch delay và retry

### 9.9. Cache trả dữ liệu cũ

Xem:

- `_rag_cache` trong `quota_guard.py`
- `cache_namespace` trong `src/rag_chain.py:445`
- nút clear cache ở sidebar `app.py`

Nếu đổi prompt/schema mà response vẫn cũ, có thể phải:

- đổi `RAG_CACHE_NAMESPACE`
- hoặc clear cache hiện tại

## 10. Khi cần bổ sung tính năng thì sửa chỗ nào

### 10.1. Thêm tài liệu mới

1. Bỏ file vào `data/`
2. Chạy lại:

```bash
python src/ingest.py
```

Nếu tài liệu có format đặc biệt, nên sửa thêm `split_documents()`.

### 10.2. Đổi model text

Nếu chỉ đổi Gemini model:

- sửa `.env` qua `GEMINI_TEXT_MODEL`

Nếu thêm provider mới:

1. thêm hàm `_try_xxx()` trong `src/llm_router.py`
2. thêm vào `providers` của `get_llm()`
3. cân nhắc quota/counter nếu provider mới cũng cần theo dõi

### 10.3. Thêm field mới cho câu trả lời RAG

Ví dụ thêm `answer_type`, `next_steps`, `policy_code`.

Phải sửa đồng bộ:

1. `SYSTEM_PROMPT`
2. `RAGResponseSchema`
3. `_normalize_rag_payload()`
4. `_parse_json_response()`
5. `_render_rag_response()`

### 10.4. Muốn chatbot nhớ hội thoại nhiều lượt

Hiện tại hệ thống gần như là hỏi độc lập từng câu, chưa có memory hội thoại thật sự trong chain.

Muốn mở rộng thì thường phải sửa ở:

- `app.py` để truyền lịch sử chat
- `src/rag_chain.py` để nhét history vào prompt

Lưu ý: làm vậy sẽ tăng token, tăng rủi ro lệch ngữ cảnh, và phải xem lại quota.

### 10.5. Muốn thay vector DB

Hiện tại đang dùng `Chroma`.

Nếu chuyển sang FAISS, Qdrant, Milvus... thì chủ yếu sửa ở:

- `src/ingest.py`
- `src/rag_chain.py`

Cụ thể là các đoạn:

- `build_vector_store()`
- `load_vector_store()`
- các hàm retrieve có score

## 11. Phần multimodal nằm ở đâu

File chính: `src/multimodal.py`

Phần này không phải lõi LangChain, nhưng vẫn là nửa còn lại của hệ thống.

Nó xử lý:

- ảnh: visual reasoning
- video/audio: Gemini File API
- output JSON có cấu trúc
- cache + quota tương tự RAG

Nếu lỗi liên quan đến ảnh/video/audio thì đi thẳng vào `src/multimodal.py`, không cần dò `src/rag_chain.py`.

## 12. Lệnh hay dùng khi bảo trì

### Cài thư viện

```bash
pip install -r requirements.txt
```

### Nạp lại dữ liệu

```bash
python src/ingest.py
```

### Xóa DB vector để ingest lại từ đầu

Trong PowerShell, đứng tại thư mục project:

```powershell
Remove-Item -LiteralPath .\chroma_db -Recurse -Force
python src/ingest.py
```

Nếu muốn reset luôn trạng thái quota theo ngày để test sạch hơn:

```powershell
Remove-Item -LiteralPath .\.quota_usage.json -Force
```

### Chạy app

```bash
streamlit run app.py
```

### Kiểm tra nhanh chunking

```bash
python check_chunks.py
```

## 13. Gợi ý cách debug nhanh theo thứ tự

Khi chatbot trả lời sai, nên debug theo thứ tự này:

1. `data/` có đúng tài liệu chưa
2. `chroma_db/` đã build lại chưa
3. `split_documents()` có chia chunk hợp lý không
4. retrieval có kéo đúng chunk không
5. `SYSTEM_PROMPT` có đủ rõ không
6. model có bị fallback không
7. parser JSON có làm rơi mất field không
8. UI có render sai không

Đây là thứ tự hợp lý vì lỗi RAG đa số nằm ở:

- dữ liệu
- chunking
- retrieval

nhiều hơn là nằm ở model.

## 14. Tóm tắt ngắn gọn để nhớ

Nếu cần nhớ thật ngắn:

- `app.py` = giao diện + gọi hệ thống
- `src/ingest.py` = biến tài liệu thành vector
- `src/rag_chain.py` = não chính của chatbot tài liệu
- `src/llm_router.py` = chọn model nào để trả lời
- `quota_guard.py` = chống 429, cache, quota
- `src/multimodal.py` = ảnh/video/audio

Trong toàn project, file đáng đọc kỹ nhất để sửa phần LangChain/RAG là:

- `src/rag_chain.py`
- `src/ingest.py`

Nếu sau này cần refactor lớn, nên bắt đầu từ 2 file này trước.
