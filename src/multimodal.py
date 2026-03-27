# =============================================================
# src/multimodal.py - PHÂN TÍCH ĐA PHƯƠNG THỨC (MỨC 2)
# =============================================================
# Kỹ thuật Mức 2:
#   1. Visual Reasoning  : Suy luận logic từ ảnh (không chỉ nhận diện)
#   2. Gemini File API   : Upload video/audio lên server Google
#   3. Structured Output : Tất cả kết quả trả về JSON chuẩn
#
# Tối ưu quota tích hợp:
#   - VisionCache        : cùng ảnh + cùng câu hỏi = 0 token
#   - RateLimiter 14 RPM : không vượt quota Vision (dùng chung LLM quota)
#   - Retry backoff      : tự phục hồi khi 429 tạm thời
# =============================================================

import os
import sys
import base64
import json
import re
import time
import hashlib
from typing import Dict, Any
from dotenv import load_dotenv
from PIL import Image
import io

import google.generativeai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from quota_guard import get_vision_cache, get_vision_limiter, get_counter, with_retry
    _vision_cache   = get_vision_cache()
    _vision_limiter = get_vision_limiter()
    _counter        = get_counter()
    _with_retry     = with_retry
except ImportError:
    _vision_cache = _vision_limiter = _counter = None
    def _with_retry(max_retries=3, base_delay=2.0):
        def decorator(func): return func
        return decorator

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# =============================================================
# SYSTEM PROMPTS
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
# PHÂN TÍCH ẢNH (Visual Reasoning)
# =============================================================
def analyze_image(image_bytes: bytes, question: str) -> Dict[str, Any]:
    """
    Phân tích ảnh với Visual Reasoning + cache + rate limit.

    Cache key = MD5(image_bytes) + question
    → Cùng ảnh + cùng câu hỏi → trả ngay kết quả cũ (0 token)
    → Cùng ảnh + câu hỏi khác → gọi API mới
    """
    # ── Cache check ──
    if _vision_cache:
        img_hash = hashlib.md5(image_bytes).hexdigest()
        cached   = _vision_cache.get(img_hash, question)
        if cached:
            if _counter:
                _counter.increment("cache_hits")
            result = dict(cached)
            result["from_cache"] = True
            return result

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Thiếu GOOGLE_API_KEY trong file .env")

    genai.configure(api_key=GOOGLE_API_KEY)

    # ── Resize ảnh → tiết kiệm token ──
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.size[0] > 1024 or pil_image.size[1] > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        img_format = pil_image.format if pil_image.format else "JPEG"
        if img_format.upper() not in ["JPEG", "PNG", "WEBP"]:
            img_format = "JPEG"

        buf = io.BytesIO()
        pil_image.save(buf, format=img_format)
        processed_bytes = buf.getvalue()
    except Exception as e:
        return _error_response("image", f"Không đọc được ảnh: {str(e)}")

    # ── Gọi Gemini Vision ──
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=IMAGE_SYSTEM_PROMPT,
        )
        prompt_parts = [
            f"Câu hỏi của sinh viên: {question}\n\nHãy phân tích ảnh sau và trả lời:",
            {"mime_type": f"image/{img_format.lower()}", "data": base64.b64encode(processed_bytes).decode("utf-8")},
        ]
        gen_config = genai.GenerationConfig(temperature=0.1, max_output_tokens=1024)

        @_with_retry(max_retries=3, base_delay=2.0)
        def _call():
            return model.generate_content(prompt_parts, generation_config=gen_config)

        if _vision_limiter:
            with _vision_limiter:
                response = _call()
        else:
            response = _call()

        if _counter:
            _counter.increment("gemini_vision")

        result               = _parse_image_response(response.text)
        result["from_cache"] = False

        # ── Lưu cache ──
        if _vision_cache:
            img_hash = hashlib.md5(image_bytes).hexdigest()
            _vision_cache.set(result, img_hash, question)

        return result

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return _error_response("quota",
                "Hết quota Gemini Vision. Chờ 1 phút rồi thử lại (reset lúc 07:00 VN).")
        return _error_response("api", f"Lỗi Gemini API: {err[:200]}")


# =============================================================
# PHÂN TÍCH VIDEO/AUDIO (Gemini File API)
# =============================================================
def analyze_media_file(file_bytes: bytes, filename: str, question: str, mime_type: str) -> Dict[str, Any]:
    """
    Phân tích video/audio với Gemini File API.

    Tại sao dùng File API (Mức 2 yêu cầu):
      - File lớn không thể gửi qua base64 → tốn băng thông
      - File API: upload 1 lần, Google lưu 48h → gọi Gemini với URI
      - Tiết kiệm token vì không encode base64 trong payload

    Quy trình:
      1. Upload → nhận file_uri
      2. Chờ PROCESSING → ACTIVE
      3. Gọi Gemini với file URI
      4. Xóa file sau khi dùng xong
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Thiếu GOOGLE_API_KEY trong file .env")

    genai.configure(api_key=GOOGLE_API_KEY)

    import tempfile
    uploaded_file = None
    tmp_path      = None

    try:
        # ── Bước 1: Lưu tạm + Upload ──
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}", dir="/tmp") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        print(f"[File API] Uploading {filename} ({len(file_bytes)/1024/1024:.1f} MB)...")
        uploaded_file = genai.upload_file(path=tmp_path, mime_type=mime_type, display_name=filename)
        print(f"[File API] Upload xong: {uploaded_file.name}")

        # ── Bước 2: Chờ ACTIVE ──
        waited = 0
        while uploaded_file.state.name == "PROCESSING" and waited < 120:
            print(f"[File API] Đang xử lý... ({waited}s)")
            time.sleep(3)
            waited       += 3
            uploaded_file = genai.get_file(uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            return _error_response("processing", f"File không xử lý được: {uploaded_file.state.name}")

        print(f"[File API] Sẵn sàng: {uploaded_file.uri}")

        # ── Bước 3: Gọi Gemini với file URI ──
        model  = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction="Bạn là trợ lý phân tích nội dung đa phương thức cho sinh viên SGU. Trả lời bằng tiếng Việt, ngắn gọn và hữu ích.",
        )
        prompt = VIDEO_SYSTEM_PROMPT.format(question=question)

        @_with_retry(max_retries=2, base_delay=3.0)
        def _call():
            return model.generate_content(
                [prompt, uploaded_file],
                generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=2048),
            )

        response = _call()
        if _counter:
            _counter.increment("gemini_vision")

        return _parse_media_response(response.text)

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return _error_response("quota", "Hết quota Gemini. Reset lúc 07:00 VN.")
        return _error_response("api", f"Lỗi File API: {err[:200]}")

    finally:
        # ── Bước 4: Dọn dẹp ──
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
                print(f"[File API] Đã xóa: {uploaded_file.name}")
            except Exception:
                pass


# =============================================================
# HELPERS
# =============================================================
def _parse_image_response(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        return {
            "image_type":      data.get("image_type",      "không xác định"),
            "extracted_data":  data.get("extracted_data",  {}),
            "reasoning":       data.get("reasoning",       ""),
            "answer":          data.get("answer",          "Không thể phân tích."),
            "recommendations": data.get("recommendations", []),
            "confidence":      data.get("confidence",      "low"),
        }
    except json.JSONDecodeError:
        return {
            "image_type": "không xác định", "extracted_data": {},
            "reasoning": "", "answer": raw,
            "recommendations": ["Kết quả không phải JSON chuẩn"], "confidence": "low",
        }


def _parse_media_response(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        return {
            "content_type": data.get("content_type", "không xác định"),
            "summary":      data.get("summary",      []),
            "action_items": data.get("action_items", []),
            "key_moments":  data.get("key_moments",  []),
            "answer":       data.get("answer",       "Không thể phân tích."),
            "confidence":   data.get("confidence",   "low"),
        }
    except json.JSONDecodeError:
        return {
            "content_type": "không xác định", "summary": [raw],
            "action_items": [], "key_moments": [], "answer": raw, "confidence": "low",
        }


def _error_response(error_type: str, message: str) -> dict:
    return {
        "image_type": f"lỗi_{error_type}", "extracted_data": {},
        "reasoning": "", "answer": message,
        "recommendations": [], "confidence": "low",
    }


def get_image_dimensions(image_bytes: bytes) -> tuple:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.size
    except Exception:
        return (0, 0)


def get_supported_media_types() -> dict:
    return {
        "image": ["image/jpeg", "image/png", "image/webp", "image/gif"],
        "video": ["video/mp4",  "video/mpeg", "video/mov", "video/avi", "video/webm"],
        "audio": ["audio/mpeg", "audio/mp3",  "audio/wav", "audio/ogg", "audio/m4a"],
    }