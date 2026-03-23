# =============================================================
# multimodal.py - PHÂN TÍCH ĐA PHƯƠNG THỨC VỚI GEMINI (LEVEL 2)
# =============================================================
# Mức 2 yêu cầu:
#   1. Visual Reasoning: Suy luận logic từ ảnh (không chỉ nhận diện)
#   2. Gemini File API: Upload file lớn (video/audio) lên server Google
#   3. Structured Extraction: Trích xuất thông tin ra JSON chuẩn
#
# Use cases trong bối cảnh SGU:
#   - Ảnh: Phân tích hóa đơn học phí, bảng điểm, lịch thi, thông báo
#   - Video/Audio: Tóm tắt nội dung buổi học, cuộc họp
#
# Lý do dùng Gemini File API thay vì base64:
#   - File lớn (video, audio) không thể gửi qua base64 → tốn băng thông
#   - File API upload 1 lần, dùng nhiều lần → tiết kiệm token
#   - Google lưu file 48 giờ trên server → không cần re-upload
#
# Structured Output: Tất cả trả về JSON chuẩn
# =============================================================

import os        # Đọc biến môi trường
import base64    # Encode ảnh nhỏ thành base64
import json      # Parse JSON response
import re        # Regex xóa markdown
import time      # Chờ file processing
from typing import Dict, Any
from dotenv import load_dotenv
from PIL import Image   # Xử lý ảnh: validate, resize
import io               # BytesIO

# Dùng google-genai SDK mới (thay thế google-generativeai đã deprecated)
# Hỗ trợ File API, không conflict với langchain-google-genai
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# =============================================================
# SYSTEM PROMPT - VISUAL REASONING (Mức 2)
# =============================================================
# Lý do thiết kế prompt này:
# - Yêu cầu AI KHÔNG chỉ nhận diện vật thể mà phải SUY LUẬN LOGIC
# - Ví dụ: không nói "đây là bảng điểm" mà phải phân tích "môn nào cần cải thiện"
# - Buộc trả về JSON để code xử lý tiếp (Structured Output)
# - Phân biệt các loại tài liệu SGU để extract đúng thông tin
# =============================================================
IMAGE_SYSTEM_PROMPT = """Bạn là trợ lý thông minh hỗ trợ sinh viên Trường Đại học Sài Gòn (SGU).

NHIỆM VỤ VISUAL REASONING (Suy luận hình ảnh):
Không chỉ nhận diện nội dung — hãy SUY LUẬN và đưa ra NHẬN XÉT HỮU ÍCH.

Hướng dẫn theo loại tài liệu:
- HOA DON HOC PHI: Trích xuất mã SV, tên, số tiền, ngày đóng, ngân hàng. Suy luận: đã đóng đủ chưa?
- BANG DIEM: Liệt kê môn, điểm, TC. Suy luận: GPA ước tính, môn nào cần cải thiện, có nguy cơ cảnh báo không?
- LICH THI: Liệt kê môn, ngày giờ, phòng. Suy luận: môn nào thi sớm nhất, có trùng lịch không?
- THONG BAO: Tóm tắt nội dung chính, deadline, ai cần làm gì.

YÊU CẦU: Trả về JSON thuần túy (không markdown, không ```json):
{"image_type":"loai tai lieu","extracted_data":{"key":"value"},"reasoning":"suy luan logic va nhan xet","answer":"tra loi cau hoi nguoi dung","recommendations":["goi y 1","goi y 2"],"confidence":"high/medium/low"}"""

VIDEO_SYSTEM_PROMPT = """Bạn là trợ lý phân tích nội dung đa phương thức cho sinh viên SGU.

Phân tích file video/audio được cung cấp theo câu hỏi: {question}

Yêu cầu phân tích:
- Tóm tắt nội dung chính (3-5 điểm quan trọng nhất)
- Liệt kê các action items / việc cần làm (nếu có)
- Thời điểm quan trọng (nếu là video)
- Người tham gia và vai trò (nếu là cuộc họp)

Trả về JSON thuần túy:
{"content_type":"loai noi dung","summary":["diem 1","diem 2"],"action_items":[{"task":"viec lam","assignee":"nguoi thuc hien","deadline":"han chot"}],"key_moments":["thoi diem quan trong"],"answer":"tra loi cau hoi","confidence":"high/medium/low"}"""


# =============================================================
# HÀM PHÂN TÍCH ẢNH (Visual Reasoning)
# =============================================================
def analyze_image(image_bytes: bytes, question: str) -> Dict[str, Any]:
    """
    Phân tích ảnh với Visual Reasoning — suy luận logic, không chỉ nhận diện.

    Kỹ thuật Mức 2:
    - Visual Reasoning: AI suy luận từ ảnh (GPA có đủ học bổng không? Lịch thi có trùng không?)
    - Structured Extraction: Trích xuất ra JSON chuẩn
    - Dùng base64 cho ảnh nhỏ (< 20MB) — phù hợp với ảnh chụp tài liệu

    Args:
        image_bytes: Dữ liệu ảnh dạng bytes
        question:    Câu hỏi của sinh viên

    Returns:
        Dict JSON: image_type, extracted_data, reasoning, answer, recommendations, confidence
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Thiếu GOOGLE_API_KEY trong file .env")

    # --- Bước 1: Cấu hình SDK và validate ảnh ---
    genai.configure(api_key=GOOGLE_API_KEY)

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Resize nếu quá lớn → tiết kiệm token (Token Management)
        # Gemini vẫn đọc rõ ảnh 1024px, không cần gửi ảnh 4K
        max_size = (1024, 1024)
        if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Xác định format
        img_format = pil_image.format if pil_image.format else "JPEG"
        if img_format.upper() not in ["JPEG", "PNG", "WEBP"]:
            img_format = "JPEG"

        # Ghi vào buffer
        buf = io.BytesIO()
        pil_image.save(buf, format=img_format)
        processed_bytes = buf.getvalue()

    except Exception as e:
        return _error_response("image", f"Không đọc được ảnh: {str(e)}")

    # --- Bước 2: Gọi Gemini với Visual Reasoning prompt ---
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            # System instruction định danh vai trò AI (yêu cầu báo cáo)
            system_instruction=IMAGE_SYSTEM_PROMPT,
        )

        # Tạo prompt kết hợp câu hỏi + ảnh base64
        prompt_parts = [
            f"Câu hỏi của sinh viên: {question}\n\nHãy phân tích ảnh sau và trả lời:",
            {
                "mime_type": f"image/{img_format.lower()}",
                "data": base64.b64encode(processed_bytes).decode("utf-8"),
            }
        ]

        response = model.generate_content(
            prompt_parts,
            generation_config=genai.GenerationConfig(
                temperature=0.1,        # Nhất quán, ít sáng tạo → phù hợp phân tích
                max_output_tokens=1024, # Giới hạn output (Token Management)
            )
        )

        return _parse_image_response(response.text)

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return _error_response("quota", "Hết quota Gemini (1500 req/ngày). Reset lúc 07:00 VN.")
        return _error_response("api", f"Lỗi Gemini API: {err[:200]}")


# =============================================================
# HÀM PHÂN TÍCH VIDEO/AUDIO (Gemini File API)
# =============================================================
def analyze_media_file(file_bytes: bytes, filename: str, question: str, mime_type: str) -> Dict[str, Any]:
    """
    Phân tích video/audio dùng Gemini File API.

    Tại sao dùng File API thay vì base64 (Mức 2 yêu cầu):
    - File lớn (30MB video) không thể gửi qua base64 → gây lỗi hoặc tốn băng thông cực lớn
    - File API: Upload 1 lần lên Google server → Gemini đọc trực tiếp từ server
    - Google lưu file 48 giờ → có thể dùng lại nhiều lần mà không cần upload lại
    - Tiết kiệm token vì không cần encode base64 trong payload

    Quy trình File API:
      1. Upload file → nhận file_uri
      2. Chờ Google xử lý file (PROCESSING → ACTIVE)
      3. Gọi Gemini với file_uri thay vì base64
      4. Nhận kết quả JSON

    Args:
        file_bytes: Dữ liệu file dạng bytes
        filename:   Tên file gốc
        question:   Câu hỏi của người dùng
        mime_type:  MIME type (video/mp4, audio/mpeg, v.v.)

    Returns:
        Dict JSON: content_type, summary, action_items, key_moments, answer, confidence
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Thiếu GOOGLE_API_KEY trong file .env")

    genai.configure(api_key=GOOGLE_API_KEY)

    # --- Bước 1: Upload file lên Google File API ---
    # File API nhận file dưới dạng stream, không cần convert
    import tempfile
    import os

    uploaded_file = None
    try:
        # Lưu tạm ra disk để upload (File API cần file path hoặc file object)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f"_{filename}",
            dir="/tmp"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        print(f"[File API] Đang upload {filename} ({len(file_bytes)/1024/1024:.1f} MB)...")

        # Upload lên Google File API
        uploaded_file = genai.upload_file(
            path=tmp_path,
            mime_type=mime_type,
            display_name=filename,
        )

        print(f"[File API] Upload xong: {uploaded_file.name}")

        # --- Bước 2: Chờ Google xử lý file ---
        # File cần thời gian để Google transcribe/process
        # Trạng thái: PROCESSING → ACTIVE (sẵn sàng dùng)
        max_wait = 60  # Chờ tối đa 60 giây
        waited = 0
        while uploaded_file.state.name == "PROCESSING" and waited < max_wait:
            print(f"[File API] Đang xử lý... ({waited}s)")
            time.sleep(3)
            waited += 3
            # Refresh trạng thái file
            uploaded_file = genai.get_file(uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            return _error_response("processing", f"File không xử lý được: {uploaded_file.state.name}")

        print(f"[File API] File sẵn sàng: {uploaded_file.uri}")

        # --- Bước 3: Gọi Gemini với file URI ---
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction="Bạn là trợ lý phân tích nội dung đa phương thức cho sinh viên SGU. Trả lời bằng tiếng Việt, ngắn gọn và hữu ích.",
        )

        prompt = VIDEO_SYSTEM_PROMPT.format(question=question)

        response = model.generate_content(
            [prompt, uploaded_file],  # Truyền file URI trực tiếp
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2048,  # Video/audio cần output dài hơn
            )
        )

        return _parse_media_response(response.text)

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return _error_response("quota", "Hết quota Gemini. Reset lúc 07:00 VN.")
        return _error_response("api", f"Lỗi File API: {err[:200]}")

    finally:
        # Dọn dẹp file tạm và xóa file khỏi Google server
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        if uploaded_file:
            try:
                # Xóa file khỏi Google server sau khi dùng xong
                genai.delete_file(uploaded_file.name)
                print(f"[File API] Đã xóa file khỏi server: {uploaded_file.name}")
            except Exception:
                pass  # Không quan trọng nếu xóa thất bại


# =============================================================
# HELPER FUNCTIONS
# =============================================================
def _parse_image_response(raw_text: str) -> dict:
    """Parse JSON response từ Gemini Vision."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        return {
            "image_type":      data.get("image_type", "không xác định"),
            "extracted_data":  data.get("extracted_data", {}),
            "reasoning":       data.get("reasoning", ""),
            "answer":          data.get("answer", "Không thể phân tích."),
            "recommendations": data.get("recommendations", []),
            "confidence":      data.get("confidence", "low"),
        }
    except json.JSONDecodeError:
        return {
            "image_type":      "không xác định",
            "extracted_data":  {},
            "reasoning":       "",
            "answer":          raw_text,
            "recommendations": ["Kết quả không phải JSON chuẩn"],
            "confidence":      "low",
        }


def _parse_media_response(raw_text: str) -> dict:
    """Parse JSON response từ Gemini sau khi phân tích video/audio."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        return {
            "content_type":  data.get("content_type", "không xác định"),
            "summary":       data.get("summary", []),
            "action_items":  data.get("action_items", []),
            "key_moments":   data.get("key_moments", []),
            "answer":        data.get("answer", "Không thể phân tích."),
            "confidence":    data.get("confidence", "low"),
        }
    except json.JSONDecodeError:
        return {
            "content_type":  "không xác định",
            "summary":       [raw_text],
            "action_items":  [],
            "key_moments":   [],
            "answer":        raw_text,
            "confidence":    "low",
        }


def _error_response(error_type: str, message: str) -> dict:
    """Tạo response lỗi có cấu trúc JSON chuẩn (không crash app)."""
    return {
        "image_type":      f"lỗi_{error_type}",
        "extracted_data":  {},
        "reasoning":       "",
        "answer":          message,
        "recommendations": [],
        "confidence":      "low",
    }


def get_image_dimensions(image_bytes: bytes) -> tuple:
    """Lấy kích thước ảnh để hiển thị trên UI."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.size
    except Exception:
        return (0, 0)


def get_supported_media_types() -> dict:
    """Trả về danh sách MIME type được hỗ trợ bởi Gemini File API."""
    return {
        "image": ["image/jpeg", "image/png", "image/webp", "image/gif"],
        "video": ["video/mp4", "video/mpeg", "video/mov", "video/avi", "video/webm"],
        "audio": ["audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/m4a"],
    }
