# HƯỚNG DẪN CHO NGƯỜI CHỈNH UI

Tài liệu này dùng để nhắc người chỉnh giao diện không vô tình làm hỏng logic gọi API Gemini hoặc làm hệ thống bị lỗi `429 Too Many Requests`.

Mục tiêu:

- cho phép chỉnh giao diện thoải mái,
- nhưng không đụng vào logic quota, cache, retry và gọi model.

---

## 1. Nguyên tắc quan trọng nhất

Nếu bạn chỉ được giao chỉnh giao diện, hãy hiểu đơn giản như sau:

- bạn **được sửa phần hiển thị**,
- bạn **không được sửa phần gọi Gemini**,
- bạn **không được biến app thành tự gửi request khi người dùng đang gõ**.

Chỉ cần giữ đúng 3 nguyên tắc này là đã tránh được phần lớn rủi ro gây `429`.

---

## 2. Những file có thể sửa khi chỉ chỉnh UI

Bạn có thể sửa chủ yếu trong file:

- [app.py](D:\Các%20công%20nghệ%20lập%20trình%20hiện%20đại\LangChain-RAG-Chatbot-V2\app.py)

Các phần có thể sửa an toàn:

- màu sắc giao diện,
- CSS,
- icon,
- tiêu đề,
- subtitle,
- text mô tả,
- chia cột,
- khoảng cách,
- container,
- expander,
- badge hiển thị,
- cách trình bày câu trả lời,
- cách trình bày `Nguồn`,
- cách trình bày `Độ tin cậy`,
- cách trình bày `Structured Output JSON`,
- bố cục sidebar.

Nói ngắn gọn:

- sửa **cách nhìn** thì được,
- đừng sửa **cách app quyết định khi nào gọi API**.

---

## 3. Những file không nên đụng nếu không hiểu logic quota

Nếu chỉ chỉnh UI, không nên sửa các file sau:

- [quota_guard.py](D:\Các%20công%20nghệ%20lập%20trình%20hiện%20đại\LangChain-RAG-Chatbot-V2\quota_guard.py)
- [src/llm_router.py](D:\Các%20công%20nghệ%20lập%20trình%20hiện%20đại\LangChain-RAG-Chatbot-V2\src\llm_router.py)
- [src/rag_chain.py](D:\Các%20công%20nghệ%20lập%20trình%20hiện%20đại\LangChain-RAG-Chatbot-V2\src\rag_chain.py)
- [src/multimodal.py](D:\Các%20công%20nghệ%20lập%20trình%20hiện%20đại\LangChain-RAG-Chatbot-V2\src\multimodal.py)
- [src/ingest.py](D:\Các%20công%20nghệ%20lập%20trình%20hiện%20đại\LangChain-RAG-Chatbot-V2\src\ingest.py)

Lý do:

- các file này đang chứa logic chống `429`,
- logic cache,
- logic retry,
- logic soft cap theo ngày,
- logic parse output,
- logic gọi Gemini text, vision và File API,
- logic embedding và ingest dữ liệu.

Nếu sửa sai ở các file này, app có thể:

- bị gọi request quá nhiều,
- mất cache,
- dễ chạm quota,
- bị `429`,
- hoặc trả lời sai định dạng.

---

## 4. Những chỗ tuyệt đối không được sửa nếu không có người hiểu backend kiểm tra lại

### 4.1. Không đổi điều kiện submit chat

Hiện tại chat chỉ nên gọi khi người dùng **bấm nút tìm kiếm**.

Không được đổi logic thành kiểu:

- chỉ cần ô nhập có text là gọi,
- vừa gõ vừa query,
- thay đổi text input là gọi,
- render lại trang là gọi.

Vì làm vậy sẽ đốt quota rất nhanh.

### 4.2. Không gọi API trong lúc render UI

Không được đặt các lệnh như:

- `rag.query(...)`
- `analyze_image(...)`
- `analyze_media_file(...)`

ở chỗ render giao diện thông thường.

Chúng chỉ nên được gọi khi có hành động rõ ràng như:

- bấm nút,
- submit form,
- người dùng chủ động yêu cầu phân tích.

### 4.3. Không bỏ cache

Cache đang giúp:

- câu hỏi cũ không gọi lại model,
- ảnh hoặc media cũ không phân tích lại vô ích,
- giảm token,
- giảm request/phút,
- giảm nguy cơ `429`.

Nếu bỏ cache, app sẽ hao quota rất nhanh.

### 4.4. Không bỏ retry và limiter

`rate limiter` và `retry` đang là lớp bảo vệ chính chống `429`.

Không được:

- xóa limiter,
- tăng quá cao giới hạn request,
- tắt retry,
- đổi retry thành spam request liên tục.

### 4.5. Không tăng mạnh `max_output_tokens` hoặc `top_k`

Tăng các giá trị này có thể:

- làm prompt dài hơn,
- tăng token output,
- chậm hơn,
- dễ chạm hạn mức hơn.

Nếu muốn tăng, phải có người hiểu logic quota kiểm tra lại.

---

## 5. Những thứ đang bảo vệ quota trong project

### 5.1. Trong `quota_guard.py`

Hiện có:

- limiter theo RPM,
- soft cap theo ngày,
- request counter,
- daily tracker lưu vào `.quota_usage.json`,
- cache cho RAG,
- cache cho vision,
- retry với exponential backoff.

Đây là lớp quan trọng nhất để bảo vệ quota.

### 5.2. Trong `app.py`

Hiện đã sửa để:

- chat chỉ chạy khi bấm nút,
- không auto query khi đang gõ,
- UI hiển thị quota usage,
- có nút xóa cache.

### 5.3. Trong `src/rag_chain.py`

Hiện đang:

- retrieve `top_k = 4`,
- chuẩn hóa output cho UI,
- xây `source`,
- ước lượng `confidence`,
- dùng lớp gọi model đã có guard.

### 5.4. Trong `src/multimodal.py`

Hiện đang:

- có limiter cho ảnh và media,
- có cache cho file + question,
- có parser chống lỗi output xấu,
- có File API flow ổn định hơn cho media.

### 5.5. Trong `src/ingest.py`

Hiện đang:

- kiểm soát quota embedding,
- tránh ingest vô tội vạ theo ngày.

---

## 6. Những việc chỉnh UI được xem là an toàn

Bạn có thể làm các việc sau mà không ảnh hưởng logic:

- đổi màu theme,
- sửa layout đẹp hơn,
- thêm section giới thiệu,
- đổi icon,
- đổi wording nút bấm,
- chỉnh kích thước nút,
- đổi text `Câu trả lời`, `Nguồn`, `Độ tin cậy`,
- làm sidebar gọn hơn,
- làm expander đẹp hơn,
- render card đẹp hơn,
- thêm divider,
- thêm caption,
- chỉnh style hiển thị JSON.

Nếu chỉ làm những việc này thì gần như an toàn.

---

## 7. Những việc nhìn giống UI nhưng thực ra dễ phá quota

Có một số thay đổi nhìn có vẻ chỉ là UI nhưng thực ra rất nguy hiểm:

- đổi text input thành auto-submit,
- bỏ nút submit và query ngay khi nhập,
- tự động load câu trả lời mẫu mỗi lần mở trang,
- tự động phân tích lại ảnh khi rerun,
- tự động gọi lại media analysis khi render kết quả,
- tự động rebuild vector DB khi vào app.

Các thay đổi này **không phải chỉ là UI** nữa, mà đã đụng vào logic request.

---

## 8. Checklist trước khi merge một thay đổi giao diện

Trước khi chốt thay đổi UI, hãy tự kiểm tra:

1. Tôi có đụng vào `quota_guard.py` không?
2. Tôi có đụng vào `src/rag_chain.py` không?
3. Tôi có đụng vào `src/multimodal.py` không?
4. Tôi có làm chat tự query khi đang gõ không?
5. Tôi có làm ảnh/media tự phân tích lại khi rerun không?
6. Tôi có vô tình bỏ cache không?
7. Tôi có tăng request lên nhiều hơn trước không?

Nếu câu trả lời là `có` ở bất kỳ mục nào, cần kiểm tra lại với người phụ trách logic backend trước khi merge.

---

## 9. Câu nhắc ngắn gọn cho teammate hoặc AI khác

Bạn có thể copy nguyên đoạn này để nhắc người khác:

```text
Bạn chỉ được chỉnh UI trong app.py.
Không được thay đổi điều kiện submit của chat/image/media.
Không được gọi rag.query(), analyze_image(), analyze_media_file() ngoài nút bấm hoặc submit form.
Không được sửa quota_guard.py, llm_router.py, rag_chain.py, multimodal.py, ingest.py nếu không hiểu logic quota.
Phải giữ nguyên cache, limiter, retry, soft cap để tránh lỗi 429.
```

---

## 10. Kết luận

Nếu chỉ chỉnh giao diện, hãy nhớ một quy tắc rất dễ thuộc:

- sửa cách app **hiển thị** thì được,
- đừng sửa cách app **gọi Gemini**.

Phần logic quota hiện tại đã được tinh chỉnh để phù hợp với Gemini free tier. Nếu giữ nguyên lớp này, người chỉnh UI có thể thoải mái làm đẹp giao diện mà không làm hệ thống bị lỗi `429` trở lại.

