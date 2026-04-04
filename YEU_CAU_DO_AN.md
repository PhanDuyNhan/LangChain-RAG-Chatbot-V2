# HƯỚNG DẪN & YÊU CẦU ĐỒ ÁN: GEMINI API

Môn học: Các Công nghệ Lập trình Hiện đại

Đối tượng: Nhóm sinh viên đăng ký đề tài tìm hiểu và ứng dụng Google Gemini API.

## 1\. TÀI LIỆU THAM KHẢO CỐT LÕI

Nhóm bắt buộc phải nghiên cứu và áp dụng các kỹ thuật từ 2 nguồn chính thống sau:

- **Gemini API Cookbook (Web):** <https://ai.google.dev/gemini-api/cookbook>
- **Gemini Cookbook (GitHub):** <https://github.com/google-gemini/cookbook>

Có thể tham khảo thêm nhưng đây là tài liệu tương đối chuẩn

**Lưu ý:** Báo cáo cần chỉ rõ nhóm đã áp dụng kỹ thuật nào từ Cookbook (Ví dụ: _Sử dụng kỹ thuật 'JSON Mode' trong cookbook để trích xuất dữ liệu_).

## 2\. CÁC CẤP ĐỘ CỦA ĐỒ ÁN (LEVELING)

Để tránh việc nhóm chỉ làm một ứng dụng chat đơn giản, đồ án cần đạt được các cấp độ sau. **Khuyến khích sinh viên nhắm tới Mức 3 và Mức 4.**

### ❌ Mức 0: Chatbot cơ bản (Không đạt)

- **Mô tả:** Giao diện có 1 ô nhập liệu -> Gửi text lên Gemini -> Hiển thị text trả về.
- **Đánh giá:** Chỉ mất 15 phút để làm. Không có giá trị kỹ thuật.

### ✅ Mức 1: RAG (Retrieval-Augmented Generation) - Chat với dữ liệu riêng

- **Mô tả:** Xây dựng hệ thống cho phép AI trả lời câu hỏi dựa trên kho dữ liệu riêng (PDF, Docx, TXT) của doanh nghiệp/cá nhân mà Google chưa từng học.
- **Bài toán:** Làm sao để AI không trả lời bịa (hallucination) và chỉ dùng thông tin trong tài liệu cung cấp?
- **Yêu cầu kỹ thuật chi tiết:**
  - **Data Pipeline:** Viết script đọc file -> Làm sạch text (Data Cleaning) -> Chia nhỏ văn bản (Chunking). _Lưu ý: Phải giải thích được chiến lược chunking (chia theo đoạn, theo trang hay theo ngữ nghĩa)._
  - **Vector Search:** Sử dụng Embeddings API để chuyển text thành vector và lưu vào Vector Database (ChromaDB, Pinecone...).
  - **Context Injection:** Khi user hỏi, hệ thống phải tìm 3-5 đoạn văn bản liên quan nhất, ghép vào Prompt gửi cho Gemini.
  - **Citations (Trích dẫn):** Câu trả lời của AI phải chỉ rõ thông tin lấy từ trang nào, tài liệu nào.
- **Ví dụ:** Chatbot Quy chế Đào tạo (Input: File PDF "Sổ tay sinh viên").
  - _User:_ "Học cải thiện điểm tốn bao nhiêu tiền?"
  - _AI:_ "Theo mục 5 trang 12 Sổ tay sinh viên, học phí cải thiện là..."

### ✅ Mức 2: Multimodal (Đa phương thức) - Thế mạnh của Gemini

- **Mô tả:** Tận dụng khả năng thấu hiểu đồng thời Văn bản, Hình ảnh, Âm thanh và Video của Gemini (điều mà các model cũ không làm được).
- **Bài toán:** Xử lý các dữ liệu phi cấu trúc trong thực tế (Hóa đơn viết tay, Video camera giám sát, File ghi âm cuộc họp).
- **Yêu cầu kỹ thuật chi tiết:**
  - **Visual Reasoning (Suy luận hình ảnh):** Không chỉ nhận diện vật thể (Object detection), mà phải suy luận logic. _Ví dụ: Không chỉ nhận diện "đây là tủ lạnh", mà phải trả lời "trong tủ lạnh còn thiếu nguyên liệu gì để nấu món phở"._
  - **Gemini File API:** Sử dụng API để upload các file media lớn (Video/Audio) lên server Google để xử lý (thay vì gửi base64 image trực tiếp gây tốn băng thông).
  - **Structured Extraction:** Trích xuất thông tin từ ảnh/video ra dạng JSON.
- **Ví dụ:**
  - _Trợ lý Dinh dưỡng:_ Chụp ảnh mâm cơm -> AI phân tích các món ăn -> Tính toán lượng Calo -> Cảnh báo nếu nhiều dầu mỡ.
  - _Trợ lý Cuộc họp:_ Tải lên file ghi âm 30 phút -> AI tóm tắt nội dung chính + Liệt kê danh sách "To-do list" cho từng người dưới dạng bảng.

### ✅ Mức 3: Autonomous Agents (Tác nhân tự chủ)

- **Mô tả:** AI không chỉ trả lời mà phải **tự suy luận nhiều bước** để hoàn thành nhiệm vụ phức tạp.
- **Yêu cầu kỹ thuật:** Sử dụng **Function Calling** kết hợp với tư duy **ReAct (Reason + Act)**.
- **Ví dụ:**
  - _User:_ "Tìm vé máy bay rẻ nhất đi Đà Nẵng tuần sau và gửi mail báo giá cho tôi."
  - _AI Agent (tự suy nghĩ):_
    - Gọi hàm search_flights(destination="Da Nang", date="next_week").
    - Nhận kết quả, phân tích tìm vé rẻ nhất.
    - Gọi hàm send_email(to="<user@mail.com>", body="...").
  - _Kết quả:_ AI tự thực hiện chuỗi hành động, con người không cần can thiệp từng bước.

### 🚀 Mức 4

_Đây là mức độ thách thức nhất, yêu cầu tối ưu hóa chi phí và hiệu năng cho bài toán lớn._

- **Context Caching (Bộ nhớ đệm ngữ cảnh):**
  - _Bài toán:_ Khi làm việc với tài liệu cực lớn (ví dụ: Bộ luật hình sự, Tài liệu kỹ thuật xe hơi dài 1000 trang), việc gửi đi gửi lại toàn bộ tài liệu mỗi lần chat rất tốn tiền và chậm.
  - _Yêu cầu:_ Sử dụng tính năng **Context Caching** của Gemini để lưu trữ context này trên server Google, giúp các request sau chạy cực nhanh và rẻ.
- **Multimodal RAG (Tìm kiếm trong Video/Ảnh):**
  - _Bài toán:_ User có kho lưu trữ 100 video bài giảng. User hỏi: _"Thầy nói về định luật Newton ở phút thứ mấy trong video nào?"_
  - _Yêu cầu:_ Hệ thống phải tìm đúng đoạn video đó và trích xuất ra. (Kết hợp Vector Database lưu trữ Embedding của Video).
- **Automated Evaluation (AI chấm điểm AI):**
  - _Bài toán:_ Làm sao biết Bot trả lời đúng hay sai?
  - _Yêu cầu:_ Xây dựng bộ quy trình "LLM-as-a-judge". Dùng một model mạnh (Gemini Pro) để chấm điểm câu trả lời của model yếu hơn (Gemini Flash) dựa trên các tiêu chí: Độ chính xác, Độ liên quan, Không chứa nội dung độc hại.

## 3\. YÊU CẦU TRỌNG TÂM CẦN CÓ TRONG BÁO CÁO

Dù chọn ý tưởng nào, nhóm cũng phải thể hiện được các kỹ thuật sau trong báo cáo và demo:

- **System Instructions (System Prompt):**
  - Định danh cho AI (Ví dụ: _"Bạn là chuyên gia y tế, chỉ trả lời dựa trên tài liệu được cung cấp, không đưa ra lời khuyên cá nhân..."_).
  - Nhóm phải show được prompt này và giải thích tại sao viết như vậy.
- **Structured Output (JSON Mode):**
  - Kết quả trả về từ Gemini KHÔNG ĐƯỢC là văn bản tự do, mà phải là **JSON chuẩn** để code có thể xử lý tiếp (lưu DB, hiển thị lên Web).
  - _Ví dụ:_ Thay vì trả về "Món này 50k", phải trả về {"dish": "Phở", "price": 50000, "currency": "VND"}.
- **Token Management & Cost Optimization:**
  - Phân tích chi phí: Mỗi request tốn bao nhiêu token?
  - Giải pháp tối ưu: Làm sao để không bị quá giới hạn (Rate limit)? Sử dụng model nào (Pro hay Flash) cho tác vụ nào để tiết kiệm tiền?

## 4\. GỢI Ý CÔNG NGHỆ ĐI KÈM (STACK)

Để làm đồ án này, nhóm nên kết hợp Gemini API với:

- **Frontend:** **Streamlit** (Python) hoặc **Vercel AI SDK** (nếu dùng Next.js). Khuyên dùng Streamlit cho nhanh gọn.
- **Vector Database (nếu làm RAG):** ChromaDB (Local).
- **Framework:** **LangChain** (giúp kết nối Gemini với các thành phần khác dễ hơn).

5\. Kiến thức cần nắm

**Mô tả:** Sử dụng trực tiếp SDK của Google để xây dựng ứng dụng AI Native.

- **Kiến thức nền cần có:** Python hoặc Node.js, hiểu về HTTP Request/JSON.
- **Kiến thức yêu cầu cần nắm trong môn (Technical Checklist):**
  - **Prompt Engineering nâng cao:** System Instructions, Few-shot prompting.
  - **Structured Output (JSON Mode):** Kỹ thuật ép AI trả về dữ liệu JSON chuẩn để lưu vào Database (thay vì trả về văn bản tự do).
  - **Function Calling (Tool Use):** Kỹ thuật để AI tự động gọi API bên ngoài (ví dụ: tự tra cứu thời tiết, tự gửi email).
  - **Multimodal Processing:** Sử dụng **File API** để xử lý video, âm thanh, PDF số lượng lớn (Context Caching).
  - **RAG Pipeline:** Embeddings & Vector Search (nếu làm Chatbot tài liệu).
- **Gợi ý đồ án:** Trợ lý phân tích Video (Tóm tắt + Quiz), Trợ lý Kế toán (Đọc hóa đơn đỏ -> Excel)