# =============================================================
# src/multimodal.py - PHÂN TÍCH ĐA PHƯƠNG THỨC (MỨC 2)
# =============================================================
# Kỹ thuật Mức 2:
#   1. Visual Reasoning  : Suy luận logic từ ảnh (không chỉ nhận diện)
#   2. Gemini File API   : Upload video/audio lên server Google
#   3. Structured Output : Tất cả kết quả trả về JSON chuẩn
#
# ĐÃ SỬA:
#   - Migrate google.generativeai (deprecated) → google.genai (SDK mới)
#   - Cú pháp mới: genai.Client(), client.models.generate_content()
#   - File API mới: client.files.upload(), client.files.delete()
#   - Giữ nguyên toàn bộ logic quota guard (cache, limiter, retry)
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

# SDK MỚI: google.genai thay thế google.generativeai (đã deprecated)
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from quota_guard import (
        get_vision_cache,
        get_vision_limiter,
        get_counter,
        get_daily_tracker,
        with_retry,
        GEMINI_TEXT_MODEL,
        GEMINI_VISION_RPD_SOFT,
    )
    _vision_cache   = get_vision_cache()
    _vision_limiter = get_vision_limiter()
    _counter        = get_counter()
    _daily_tracker  = get_daily_tracker()
    _with_retry     = with_retry
except ImportError:
    _vision_cache = _vision_limiter = _counter = _daily_tracker = None
    GEMINI_TEXT_MODEL = "gemini-2.5-flash"
    GEMINI_VISION_RPD_SOFT = 18
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

YÊU CẦU:
- Trả lời NGẮN GỌN, ưu tiên đúng trọng tâm câu hỏi người dùng.
- Với lịch thi/bảng điểm dài, chỉ nêu tối đa 5 dòng quan trọng nhất.
- Trả về JSON thuần túy (không markdown, không ```json):
{"image_type":"loai tai lieu","extracted_data":{"key":"value"},"reasoning":"suy luan logic va nhan xet","answer":"tra loi cau hoi nguoi dung","recommendations":["goi y 1","goi y 2"],"confidence":"high/medium/low"}"""

VIDEO_SYSTEM_PROMPT = """Bạn là trợ lý phân tích nội dung đa phương thức cho sinh viên SGU.

Phân tích file video/audio được cung cấp theo câu hỏi: {question}

Yêu cầu phân tích:
- Tóm tắt nội dung chính (3-5 điểm quan trọng nhất)
- Liệt kê các action items / việc cần làm (nếu có)
- Thời điểm quan trọng (nếu là video)
- Người tham gia và vai trò (nếu là cuộc họp)

Trả về JSON thuần túy, ngắn gọn:
{"content_type":"loai noi dung","summary":["diem 1","diem 2"],"action_items":[{"task":"viec lam","assignee":"nguoi thuc hien","deadline":"han chot"}],"key_moments":["thoi diem quan trong"],"answer":"tra loi cau hoi","confidence":"high/medium/low"}"""

def _strip_markdown_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    return text

def _extract_balanced_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return ""


def _extract_json_fragment(text: str):
    cleaned = _strip_markdown_json(text)
    decoder = json.JSONDecoder()
    for candidate in [cleaned, _extract_balanced_json(cleaned)]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
        if "{" in candidate:
            try:
                obj, _ = decoder.raw_decode(candidate[candidate.find("{"):])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


def _extract_string_field(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
    if not match:
        return ""
    return bytes(match.group(1), "utf-8").decode("unicode_escape").strip()


def _extract_array_of_strings(text: str, key: str) -> list:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    inner = match.group(1)
    return [
        bytes(item, "utf-8").decode("unicode_escape").strip()
        for item in re.findall(r'"((?:\\.|[^"\\])*)"', inner)
        if item.strip()
    ]


# =============================================================
# KHỞI TẠO CLIENT (SDK MỚI)
# =============================================================
def _get_client() -> genai.Client:
    """Tạo Gemini client với SDK mới google.genai."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Thiếu GOOGLE_API_KEY trong file .env")
    return genai.Client(api_key=GOOGLE_API_KEY)


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

    client = _get_client()

    # ── Resize ảnh → tiết kiệm token ──
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.size[0] > 1024 or pil_image.size[1] > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        img_format = pil_image.format if pil_image.format else "JPEG"
        if img_format.upper() not in ["JPEG", "PNG", "WEBP"]:
            img_format = "JPEG"
        # Chuyển RGBA → RGB nếu cần (JPEG không hỗ trợ RGBA)
        if img_format.upper() == "JPEG" and pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")

        buf = io.BytesIO()
        pil_image.save(buf, format=img_format)
        processed_bytes = buf.getvalue()
    except Exception as e:
        return _error_response("image", f"Không đọc được ảnh: {str(e)}")

    # ── Gọi Gemini Vision (SDK mới) ──
    try:
        prompt_text = f"Câu hỏi của sinh viên: {question}\n\nHãy phân tích ảnh sau và trả lời:"

        # SDK mới: dùng types.Part để truyền ảnh inline
        image_part = types.Part.from_bytes(
            data=processed_bytes,
            mime_type=f"image/{img_format.lower()}",
        )

        @_with_retry(max_retries=3, base_delay=2.0)
        def _call():
            return client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt_text),
                            image_part,
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=IMAGE_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )

        if _daily_tracker and not _daily_tracker.can_consume("gemini_vision", GEMINI_VISION_RPD_SOFT):
            return _error_response(
                "quota",
                "Đã chạm soft cap Gemini trong ngày để tránh hết quota free. Hãy thử lại vào ngày mới hoặc dùng fallback."
            )

        if _vision_limiter:
            with _vision_limiter:
                response = _call()
        else:
            response = _call()

        if _counter:
            _counter.increment("gemini_vision")
        if _daily_tracker:
            _daily_tracker.increment("gemini_vision")

        result               = _parse_image_response(response.text)
        result["from_cache"] = False

        # ── Lưu cache ──
        if _vision_cache:
            img_hash = hashlib.md5(image_bytes).hexdigest()
            _vision_cache.set(result, img_hash, question)

        return result

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            return _error_response(
                "quota",
                "Hết quota Gemini Vision. Chờ 1 phút rồi thử lại (reset lúc 07:00 VN)."
            )
        return _error_response("api", f"Lỗi Gemini API: {err[:200]}")


# =============================================================
# PHÂN TÍCH VIDEO/AUDIO (Gemini File API - SDK mới)
# =============================================================
def analyze_media_file(
    file_bytes: bytes, filename: str, question: str, mime_type: str
) -> Dict[str, Any]:
    """
    Phân tích video/audio với Gemini File API (SDK mới google.genai).

    Tại sao dùng File API (Mức 2 yêu cầu):
      - File lớn không thể gửi qua base64 → tốn băng thông
      - File API: upload 1 lần, Google lưu 48h → gọi Gemini với URI
      - Tiết kiệm token vì không encode base64 trong payload

    Quy trình:
      1. Upload → nhận file object (có uri)
      2. Chờ state ACTIVE
      3. Gọi Gemini với file part
      4. Xóa file sau khi dùng xong
    """
    client        = _get_client()
    uploaded_file = None
    tmp_path      = None
    file_hash     = hashlib.md5(file_bytes).hexdigest()

    if _vision_cache:
        cached = _vision_cache.get("media", file_hash, question)
        if cached:
            if _counter:
                _counter.increment("cache_hits")
            result = dict(cached)
            result["from_cache"] = True
            return result

    try:
        import tempfile

        # ── Bước 1: Lưu tạm + Upload ──
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{filename}"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        print(f"[File API] Uploading {filename} ({len(file_bytes)/1024/1024:.1f} MB)...")

        # SDK mới: client.files.upload()
        with open(tmp_path, "rb") as f:
            uploaded_file = client.files.upload(
                file=f,
                config=types.UploadFileConfig(
                    mime_type=mime_type,
                    display_name=filename,
                ),
            )
        print(f"[File API] Upload xong: {uploaded_file.name}")

        # ── Bước 2: Chờ ACTIVE ──
        waited = 0
        while uploaded_file.state.name == "PROCESSING" and waited < 120:
            print(f"[File API] Đang xử lý... ({waited}s)")
            time.sleep(3)
            waited       += 3
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            return _error_response(
                "processing",
                f"File không xử lý được: {uploaded_file.state.name}"
            )

        print(f"[File API] Sẵn sàng: {uploaded_file.uri}")

        # ── Bước 3: Gọi Gemini với file part (SDK mới) ──
        prompt        = VIDEO_SYSTEM_PROMPT.format(question=question)
        file_part     = types.Part.from_uri(
            file_uri=uploaded_file.uri,
            mime_type=mime_type,
        )

        @_with_retry(max_retries=2, base_delay=3.0)
        def _call():
            return client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            file_part,
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "Bạn là trợ lý phân tích nội dung đa phương thức cho sinh viên SGU. "
                        "Trả lời bằng tiếng Việt, ngắn gọn và hữu ích."
                    ),
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )

        if _daily_tracker and not _daily_tracker.can_consume("gemini_vision", GEMINI_VISION_RPD_SOFT):
            return _error_response(
                "quota",
                "Đã chạm soft cap Gemini trong ngày để tránh hết quota free. Hãy thử lại vào ngày mới hoặc dùng fallback."
            )

        if _vision_limiter:
            with _vision_limiter:
                response = _call()
        else:
            response = _call()

        if _counter:
            _counter.increment("gemini_vision")
        if _daily_tracker:
            _daily_tracker.increment("gemini_vision")

        result = _parse_media_response(response.text)
        result["from_cache"] = False
        if _vision_cache:
            _vision_cache.set(result, "media", file_hash, question)
        return result

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
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
                client.files.delete(name=uploaded_file.name)
                print(f"[File API] Đã xóa: {uploaded_file.name}")
            except Exception:
                pass


# =============================================================
# HELPERS
# =============================================================
def _parse_image_response(raw: str) -> dict:
    text = _strip_markdown_json(raw)
    data = _extract_json_fragment(text)
    if isinstance(data, dict):
        return {
            "image_type":      data.get("image_type",      "không xác định"),
            "extracted_data":  data.get("extracted_data",  {}),
            "reasoning":       data.get("reasoning",       ""),
            "answer":          data.get("answer",          "Không thể phân tích."),
            "recommendations": data.get("recommendations", []),
            "confidence":      data.get("confidence",      "low"),
        }

    answer = _extract_string_field(text, "answer")
    reasoning = _extract_string_field(text, "reasoning")
    image_type = _extract_string_field(text, "image_type")
    confidence = _extract_string_field(text, "confidence")
    recommendations = _extract_array_of_strings(text, "recommendations")

    if answer or image_type:
        return {
            "image_type": image_type or "không xác định",
            "extracted_data": {},
            "reasoning": reasoning,
            "answer": answer or "Đã nhận diện được một phần nội dung nhưng JSON chưa hoàn chỉnh.",
            "recommendations": recommendations or ["Gemini trả JSON chưa hoàn chỉnh, nên xem lại câu trả lời tóm tắt."],
            "confidence": confidence or "medium",
        }

    return {
        "image_type": "không xác định", "extracted_data": {},
        "reasoning": "", "answer": raw,
        "recommendations": ["Kết quả không phải JSON chuẩn"], "confidence": "low",
    }


def _parse_media_response(raw: str) -> dict:
    text = _strip_markdown_json(raw)
    data = _extract_json_fragment(text)
    if isinstance(data, dict):
        return {
            "content_type": data.get("content_type", "không xác định"),
            "summary":      data.get("summary",      []),
            "action_items": data.get("action_items", []),
            "key_moments":  data.get("key_moments",  []),
            "answer":       data.get("answer",       "Không thể phân tích."),
            "confidence":   data.get("confidence",   "low"),
        }

    answer = _extract_string_field(text, "answer")
    content_type = _extract_string_field(text, "content_type")
    confidence = _extract_string_field(text, "confidence")
    summary = _extract_array_of_strings(text, "summary")
    key_moments = _extract_array_of_strings(text, "key_moments")
    if answer or summary:
        return {
            "content_type": content_type or "không xác định",
            "summary": summary,
            "action_items": [],
            "key_moments": key_moments,
            "answer": answer or "Đã trích được một phần nội dung nhưng JSON chưa hoàn chỉnh.",
            "confidence": confidence or "medium",
        }

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
