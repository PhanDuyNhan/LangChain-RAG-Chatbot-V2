# Semantic Chunking — Giải thích dễ hiểu

> Tài liệu này dành cho người **không chuyên về lập trình hay AI**.
> Nhóm 31 — Thái Minh Khang, Lê Huỳnh Trúc Vy, Trần Phước Thuận, Phan Duy Nhân.

---

## 1. Chúng tôi đang làm gì?

Nhóm đang xây dựng một **chatbot thông minh cho sinh viên SGU** — kiểu như trợ lý ảo, có thể trả lời các câu hỏi về quy chế, học phí, thủ tục… dựa trên các tài liệu chính thức của trường.

Để chatbot này "hiểu" được tài liệu, máy tính cần phải:
1. **Đọc** toàn bộ tài liệu (file PDF, Word…).
2. **Chia** tài liệu dài thành nhiều đoạn nhỏ.
3. Khi có câu hỏi, máy sẽ **tìm đoạn phù hợp nhất** trong kho tài liệu để trả lời.

Bài tập này tập trung vào **bước số 2 — chia tài liệu** sao cho hợp lý.

---

## 2. Tại sao phải chia tài liệu?

Hãy tưởng tượng bạn có 1 **quyển cẩm nang sinh viên dày 200 trang**. Nếu đưa nguyên cuốn cho máy đọc để tìm câu trả lời, máy sẽ "choáng" — vì lượng thông tin quá nhiều, không biết chỗ nào liên quan đến câu hỏi.

→ Giải pháp: **cắt nhỏ** cuốn sách thành các **đoạn** (gọi là *chunks*), mỗi đoạn chứa 1 nội dung tương đối độc lập. Khi có câu hỏi, máy chỉ cần tìm đúng 1-2 đoạn liên quan thay vì đọc cả cuốn.

**Vấn đề đặt ra:** Cắt ở đâu? Cắt như thế nào để không làm **đứt mạch ý nghĩa**?

---

## 3. Hai cách cắt tài liệu

### Cách cũ — Cắt theo độ dài cố định
Giống như bạn cầm thước chia quyển sách thành các đoạn **mỗi đoạn 1000 chữ**, bất kể nội dung đang nói về cái gì.

- ✅ **Ưu điểm:** Nhanh, dễ làm, miễn phí.
- ❌ **Nhược điểm:** Có thể **cắt giữa câu**, **giữa ý**. Ví dụ: 1 đoạn văn đang nói về học phí bị cắt đôi, nửa trước rơi vào chunk A, nửa sau rơi vào chunk B → máy trả lời sai.

### Cách mới — Cắt theo ý nghĩa (**Semantic Chunking**)
Đây là cách **thông minh** hơn. Máy sẽ:
1. Đọc từng câu trong tài liệu.
2. **Hiểu nội dung** mỗi câu đang nói về chủ đề gì.
3. **Chỉ cắt** tại những chỗ **chuyển sang chủ đề khác**.

Giống như 1 **biên tập viên** đọc bài báo và đánh dấu chỗ chuyển đoạn — khác với việc bạn dùng thước đo 10cm rồi cắt cơ học.

- ✅ **Ưu điểm:** Mỗi đoạn giữ nguyên 1 chủ đề → máy trả lời chính xác hơn.
- ❌ **Nhược điểm:** Chậm hơn, tốn "công" máy hơn (vì máy phải đọc hiểu từng câu).

---

## 4. "Ngưỡng" là gì? — Tham số quan trọng nhất

Khi cắt theo ý nghĩa, máy cần **1 tiêu chuẩn** để quyết định: *"Khoảng cách ý nghĩa giữa 2 câu phải KHÁC BIỆT đến mức nào thì mới cắt?"*

Tiêu chuẩn đó gọi là **ngưỡng** (`breakpoint_threshold_amount`), có giá trị từ **0 đến 100**:

| Ngưỡng | Ý nghĩa |
|:---:|---|
| **Cao** (vd: 95) | Máy **khắt khe** — chỉ cắt khi 2 câu có chủ đề **rất khác nhau** (đổi hẳn chủ đề lớn). → Tạo ra **ít đoạn, mỗi đoạn dài**. |
| **Thấp** (vd: 60) | Máy **dễ dãi** hơn — cắt cả khi 2 câu chỉ **hơi khác ý** một chút. → Tạo ra **nhiều đoạn, mỗi đoạn ngắn**. |

**Ẩn dụ dễ hiểu:** Ngưỡng giống như **"độ nhạy" của máy phát hiện chuyển đoạn**.
- Ngưỡng cao = máy lười, chỉ báo động khi có thay đổi cực lớn.
- Ngưỡng thấp = máy nhạy, báo động cả khi thay đổi nhỏ.

---

## 5. Thí nghiệm của nhóm

### Đoạn văn bản thử nghiệm

Nhóm dùng 1 đoạn văn bản **cố ý chứa 2 chủ đề khác nhau**:

> *"LangChain là một framework mã nguồn mở giúp đơn giản hóa việc xây dựng các ứng dụng dựa trên mô hình ngôn ngữ lớn (LLM). Nó cung cấp các module để quản lý prompt, kết nối LLM, và xây dựng các chuỗi xử lý phức tạp. Một trong những ứng dụng phổ biến nhất là hệ thống Hỏi-Đáp Tăng cường Truy xuất (RAG). Hệ thống RAG kết hợp LLM với một cơ sở dữ liệu vector để cung cấp câu trả lời chính xác hơn. **Trong một diễn biến khác, thị trường chứng khoán hôm nay có nhiều biến động.** Chỉ số VN-Index giảm nhẹ vào cuối phiên giao dịch buổi sáng. Các nhà đầu tư đang tỏ ra thận trọng trước các thông tin vĩ mô."*

Đoạn này có **2 chủ đề rõ ràng**:
- **Phần đầu** (4 câu): nói về **công nghệ AI** (LangChain, RAG).
- **Phần sau** (3 câu): nói về **thị trường chứng khoán**.

→ Bằng mắt thường, 1 người đọc sẽ thấy điểm "chuyển chủ đề" rõ nhất là ở câu *"Trong một diễn biến khác, thị trường chứng khoán…"*.

### Kết quả chạy thí nghiệm

Nhóm chạy máy với **2 cấu hình ngưỡng khác nhau**:

| Cấu hình | Ngưỡng | Số đoạn máy cắt được |
|:---:|:---:|:---:|
| **1** — Khắt khe | **95** | **2 đoạn** |
| **2** — Dễ dãi | **60** | **3 đoạn** |

### Chi tiết các đoạn

**Cấu hình 1 (ngưỡng 95):** Máy cắt ra **2 đoạn**:
- **Đoạn 1:** Toàn bộ phần nói về LangChain và RAG.
- **Đoạn 2:** Toàn bộ phần nói về chứng khoán.

**Cấu hình 2 (ngưỡng 60):** Máy cắt ra **3 đoạn**:
- **Đoạn 1:** Giới thiệu về LangChain (2 câu đầu).
- **Đoạn 2:** Phần giới thiệu RAG + câu chuyển sang chứng khoán.
- **Đoạn 3:** Phần chứng khoán còn lại.

---

## 6. Giải thích — Tại sao kết quả khác nhau?

### Với ngưỡng **95** (máy khắt khe) → chỉ 2 đoạn

Máy nhìn cả đoạn văn và chỉ thấy **1 chỗ duy nhất** có sự thay đổi chủ đề **thật sự lớn** — đó là chỗ chuyển từ "AI" sang "chứng khoán". Những chỗ khác (vd: chuyển từ "LangChain" sang "RAG") tuy có khác ý chút nhưng vẫn cùng một chủ đề lớn (đều là công nghệ AI), nên máy **không cắt**.

→ Kết quả: **2 đoạn lớn**, mỗi đoạn trọn vẹn 1 chủ đề.

### Với ngưỡng **60** (máy dễ dãi) → 3 đoạn

Máy trở nên **nhạy hơn** — không chỉ phát hiện chỗ đổi chủ đề lớn (AI → chứng khoán), mà còn phát hiện cả **chỗ đổi ý nhỏ hơn** trong nội bộ chủ đề AI (từ "giới thiệu LangChain" sang "giới thiệu RAG").

→ Kết quả: **3 đoạn nhỏ hơn**, mỗi đoạn chỉ chứa 1 ý nhỏ.

### Tóm lại

> **Ngưỡng càng cao → máy càng khắt khe → ít đoạn hơn, mỗi đoạn dài.**
> **Ngưỡng càng thấp → máy càng nhạy → nhiều đoạn hơn, mỗi đoạn ngắn.**

---

## 7. Ý nghĩa thực tiễn — Chọn ngưỡng nào cho tốt?

Không có con số "ngưỡng tốt nhất" cho mọi trường hợp. Việc chọn ngưỡng phải **tùy theo bài toán**:

| Trường hợp | Nên dùng ngưỡng | Lý do |
|---|:---:|---|
| Tóm tắt tài liệu dài | **Cao** (85-95) | Cần đoạn dài, đầy đủ ngữ cảnh |
| Chatbot Hỏi-Đáp chi tiết | **Thấp** (50-70) | Cần đoạn ngắn, dễ tìm chính xác |
| Tài liệu nhiều chủ đề xen lẫn | **Thấp** | Để tách rõ từng chủ đề |
| Tài liệu 1 chủ đề xuyên suốt | **Cao** | Tránh cắt vụn không cần thiết |

Trong dự án chatbot SGU, nhóm sẽ cần **thử nhiều giá trị** khác nhau và chọn ngưỡng cho kết quả chatbot **trả lời chính xác nhất**.

---

## 8. Kết luận

Qua bài tập này, nhóm đã:

1. **Hiểu** được vì sao việc chia tài liệu thành các đoạn nhỏ là quan trọng trong chatbot AI.
2. **Áp dụng thành công** kỹ thuật Semantic Chunking — một cách cắt tài liệu thông minh dựa trên ý nghĩa thay vì độ dài.
3. **Chứng minh bằng thực nghiệm** rằng tham số "ngưỡng" điều khiển trực tiếp số đoạn được tạo ra:
   - Ngưỡng **95** → **2 đoạn** (ít và dài).
   - Ngưỡng **60** → **3 đoạn** (nhiều và ngắn).
4. **Rút ra bài học** thực tiễn: chọn ngưỡng phải dựa vào đặc điểm tài liệu và mục đích sử dụng.

Kết quả này giúp nhóm có cơ sở để tối ưu phần chia tài liệu trong chatbot SGU sau này — đóng góp trực tiếp vào việc nâng cao độ chính xác của câu trả lời.
