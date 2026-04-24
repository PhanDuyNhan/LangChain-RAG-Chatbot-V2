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
import unicodedata
from datetime import datetime
from typing import Dict, Any, List, Optional
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
IMAGE_CACHE_NAMESPACE = "image_v4"


# =============================================================
# SYSTEM PROMPTS
# =============================================================
IMAGE_SYSTEM_PROMPT = """Bạn là trợ lý thông minh hỗ trợ sinh viên Trường Đại học Sài Gòn (SGU).

NHIỆM VỤ VISUAL REASONING:
- Không chỉ nhận diện nội dung trong ảnh.
- Phải suy luận để trả lời đúng trọng tâm câu hỏi người dùng.
- Ưu tiên câu trả lời hữu ích cho sinh viên, ngắn gọn nhưng đủ ý.
- Nếu ảnh có nhiều dòng/bảng dài, chỉ trích xuất thông tin QUAN TRỌNG NHẤT phục vụ câu hỏi.

PHÂN LOẠI VÀ TRÍCH XUẤT:
- HOA DON HOC PHI:
  extracted_data nên ngắn gọn, ví dụ: {"ma_sv":"","ho_ten":"","so_tien":"","ngay_dong":"","trang_thai":""}
- BANG DIEM:
  extracted_data nên ngắn gọn, ví dụ: {"hoc_ky":"","gpa_uoc_tinh":"","mon_noi_bat":["Mon A (8.5)","Mon B (6.0)"]}
- LICH THI:
  extracted_data nên ngắn gọn, ví dụ: {"ky_thi":"","mon_thi_dau_tien":"","ngay_thi_dau_tien":"dd/mm/yyyy","gio_thi_dau_tien":"HH:MM","phong_thi_dau_tien":"",
  "co_trung_lich":"co|khong","mon_hoc_chinh":["Mon A","Mon B"]}
- THONG BAO:
  extracted_data nên ngắn gọn, ví dụ: {"tieu_de":"","deadline":"","doi_tuong":"","yeu_cau":""}
- MAN HINH / DASHBOARD / SCREENSHOT:
  extracted_data nên ngắn gọn, ví dụ: {"tieu_de":"","chi_so_chinh":["Gemini 2.5 Flash: 0/5 RPM","Gemini Embedding: 43/100 RPM"],"trang_thai":"","ghi_chu":""}

YÊU CẦU CÂU TRẢ LỜI:
- answer phải trả lời trực diện câu hỏi người dùng trong 2-4 câu.
- Nếu là lịch thi, ưu tiên nêu môn thi sớm nhất, ngày giờ, phòng, có trùng lịch hay không nếu nhìn thấy.
- Nếu là thông báo, ưu tiên nêu deadline và việc cần làm.
- Nếu là ảnh chụp màn hình hoặc dashboard, hãy nêu những thông tin đang hiển thị rõ nhất và trả lời trực diện câu hỏi người dùng từ nội dung nhìn thấy.
- recommendations chỉ gồm tối đa 3 gợi ý hành động cụ thể.
- confidence:
  - high: đọc rõ được phần lớn trường quan trọng và suy luận nhất quán
  - medium: đọc được ý chính nhưng thiếu vài trường
  - low: ảnh mờ, thiếu dữ liệu hoặc không chắc chắn
- extracted_data không cần liệt kê toàn bộ bảng dài. Ưu tiên các trường cốt lõi nhất.

Trả về JSON thuần túy (không markdown, không ```json):
{"image_type":"loai tai lieu","extracted_data":{"key":"value"},"reasoning":"suy luan ngan gon","answer":"tra loi cau hoi nguoi dung","recommendations":["goi y 1","goi y 2"],
"confidence":"high/medium/low"}"""

VIDEO_SYSTEM_PROMPT = """Bạn là trợ lý phân tích nội dung đa phương thức cho sinh viên SGU.

Phân tích file video/audio được cung cấp theo câu hỏi: {question}

Yêu cầu phân tích:
- Tóm tắt nội dung chính (3-5 điểm quan trọng nhất)
- Liệt kê các action items / việc cần làm (nếu có)
- Thời điểm quan trọng (nếu là video)
- Người tham gia và vai trò (nếu là cuộc họp)

Trả về JSON thuần túy, ngắn gọn:
{{"content_type":"loai noi dung","summary":["diem 1","diem 2"],"action_items":[{{"task":"viec lam","assignee":"nguoi thuc hien","deadline":"han chot"}}],"key_moments":["thoi diem quan trong"],
"answer":"tra loi cau hoi","confidence":"high/medium/low"}}"""

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
    value = match.group(1)
    if "\\" not in value:
        return value.strip()
    try:
        return json.loads(f'"{value}"').strip()
    except Exception:
        try:
            return bytes(value, "utf-8").decode("unicode_escape").strip()
        except Exception:
            return value.strip()


def _extract_array_of_strings(text: str, key: str) -> list:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    inner = match.group(1)
    return [
        _extract_string_field(f'"value":"{item}"', "value")
        for item in re.findall(r'"((?:\\.|[^"\\])*)"', inner)
        if item.strip()
    ]


def _response_to_text(response: Any) -> str:
    """
    SDK mới đôi khi làm `response.text` raise lỗi dù candidate vẫn có text.
    Bóc thủ công để tránh crash ở Vision/File API.
    """
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass

    try:
        candidates = getattr(response, "candidates", None) or []
        chunks = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
        if chunks:
            return "\n".join(chunks).strip()
    except Exception:
        pass

    try:
        return str(response)
    except Exception:
        return ""


def _normalize_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_confidence(value: Any) -> str:
    normalized = _normalize_key(value)
    if normalized in {"HIGH", "MEDIUM", "LOW"}:
        return normalized.lower()
    if "CAO" in normalized:
        return "high"
    if "TRUNG BINH" in normalized or "MEDIUM" in normalized:
        return "medium"
    return "low"


def _normalize_image_type(value: Any) -> str:
    normalized = _normalize_key(value)
    if not normalized:
        return "không xác định"
    if "LICH THI" in normalized:
        return "LICH THI"
    if "THONG BAO" in normalized:
        return "THONG BAO"
    if "SCREENSHOT" in normalized or "DASHBOARD" in normalized or "MAN HINH" in normalized or "GIAO DIEN" in normalized:
        return "SCREENSHOT"
    if "HOA DON" in normalized or "BIEN LAI" in normalized or "PHIEU THU" in normalized:
        return "HOA DON HOC PHI"
    if "BANG DIEM" in normalized or "PHIEU DIEM" in normalized:
        return "BANG DIEM"
    return _clean_text(value) or "không xác định"


def _normalize_recommendations(value: Any) -> List[str]:
    if isinstance(value, list):
        items = [_clean_text(item) for item in value if _clean_text(item)]
        return items[:3]
    text = _clean_text(value)
    return [text] if text else []


def _needs_regenerated_answer(answer: Any) -> bool:
    text = _clean_text(answer)
    if not text:
        return True
    normalized = _normalize_key(text)
    placeholders = [
        "JSON CHUA HOAN CHINH",
        "KET QUA KHONG PHAI JSON CHUAN",
        "KHONG THE PHAN TICH",
        "DA NHAN DIEN DUOC MOT PHAN NOI DUNG",
        "CHUA DU DU LIEU DE TRA LOI RO RANG",
    ]
    return any(marker in normalized for marker in placeholders)


def _safe_mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick_value(data: Dict[str, Any], *candidates: str) -> Any:
    if not isinstance(data, dict):
        return ""
    normalized_data = {_normalize_key(key): value for key, value in data.items()}
    normalized_candidates = [_normalize_key(candidate) for candidate in candidates]

    for candidate in normalized_candidates:
        if candidate in normalized_data and normalized_data[candidate] not in ("", None, [], {}):
            return normalized_data[candidate]
    for candidate in normalized_candidates:
        for key, value in normalized_data.items():
            if candidate and candidate in key and value not in ("", None, [], {}):
                return value
    return ""


def _parse_datetime(date_text: str, time_text: str) -> Optional[datetime]:
    date_text = _clean_text(date_text)
    time_text = _clean_text(time_text) or "00:00"
    if not date_text:
        return None
    candidates = [
        f"{date_text} {time_text}",
        f"{date_text} 00:00",
    ]
    formats = [
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%d-%m-%y %H:%M",
    ]
    for candidate in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def _extract_object_field(text: str, key: str) -> Dict[str, Any]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\{{', text)
    if not match:
        return {}

    start = match.end() - 1
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
                fragment = text[start:idx + 1]
                try:
                    data = json.loads(fragment)
                    return data if isinstance(data, dict) else {}
                except Exception:
                    break
    return {}


def _extract_exam_entries(extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_entries = _pick_value(
        extracted_data,
        "lich_thi",
        "exam_schedule",
        "schedule",
        "entries",
        "items",
    )
    if not isinstance(raw_entries, list):
        earliest_subject = _clean_text(_pick_value(extracted_data, "mon_thi_dau_tien", "mon_dau_tien"))
        earliest_date = _clean_text(_pick_value(extracted_data, "ngay_thi_dau_tien", "ngay_dau_tien"))
        earliest_time = _clean_text(_pick_value(extracted_data, "gio_thi_dau_tien", "gio_dau_tien"))
        earliest_room = _clean_text(_pick_value(extracted_data, "phong_thi_dau_tien", "phong_dau_tien"))
        if any([earliest_subject, earliest_date, earliest_time, earliest_room]):
            return [{
                "mon_hoc": earliest_subject,
                "ngay_thi": earliest_date,
                "gio_thi": earliest_time,
                "phong": earliest_room,
                "ca_thi": "",
                "exam_at": _parse_datetime(earliest_date, earliest_time),
            }]
        return []

    entries: List[Dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        mon_hoc = _clean_text(_pick_value(item, "mon_hoc", "ten_mon", "mon", "hoc_phan", "noi_dung"))
        ngay_thi = _clean_text(_pick_value(item, "ngay_thi", "ngay", "date"))
        gio_thi = _clean_text(_pick_value(item, "gio_thi", "gio", "thoi_gian", "time"))
        phong = _clean_text(_pick_value(item, "phong", "phong_thi", "dia_diem", "room"))
        ca_thi = _clean_text(_pick_value(item, "ca_thi", "ca", "shift"))
        exam_at = _parse_datetime(ngay_thi, gio_thi)
        if mon_hoc or ngay_thi or gio_thi or phong:
            entries.append(
                {
                    "mon_hoc": mon_hoc,
                    "ngay_thi": ngay_thi,
                    "gio_thi": gio_thi,
                    "phong": phong,
                    "ca_thi": ca_thi,
                    "exam_at": exam_at,
                }
            )

    entries.sort(
        key=lambda entry: (
            entry["exam_at"] is None,
            entry["exam_at"] or datetime.max,
            entry["mon_hoc"] or "zzz",
        )
    )
    return entries


def _detect_exam_conflicts(entries: List[Dict[str, Any]]) -> int:
    seen = {}
    conflicts = 0
    for entry in entries:
        key = (_clean_text(entry.get("ngay_thi")), _clean_text(entry.get("gio_thi")))
        if not key[0]:
            continue
        seen[key] = seen.get(key, 0) + 1
    for count in seen.values():
        if count > 1:
            conflicts += count - 1
    return conflicts


def _extract_exam_conflict_flag(extracted_data: Dict[str, Any]) -> Optional[bool]:
    flag = _normalize_key(_pick_value(extracted_data, "co_trung_lich", "trung_lich", "conflict"))
    if not flag:
        return None
    if flag in {"CO", "YES", "TRUE"}:
        return True
    if flag in {"KHONG", "NO", "FALSE"}:
        return False
    return None


def _build_exam_answer(entries: List[Dict[str, Any]], question: str, fallback_answer: str) -> str:
    if not entries:
        return fallback_answer

    earliest = entries[0]
    mon_hoc = earliest.get("mon_hoc") or "môn ở dòng đầu tiên"
    ngay_thi = earliest.get("ngay_thi")
    gio_thi = earliest.get("gio_thi")
    phong = earliest.get("phong")
    conflicts = _detect_exam_conflicts(entries)
    question_key = _normalize_key(question)

    time_parts = []
    if gio_thi:
        time_parts.append(f"vào lúc {gio_thi}")
    if ngay_thi:
        time_parts.append(f"ngày {ngay_thi}")
    where_part = f" tại phòng {phong}" if phong else ""

    base_answer = f"Môn thi đầu tiên là {mon_hoc}"
    if time_parts:
        base_answer += " " + " ".join(time_parts)
    base_answer += where_part + "."

    if "CHUAN BI" in question_key or "CAN GI" in question_key or "CAN CHUAN BI" in question_key:
        base_answer += f" Bạn nên ưu tiên ôn môn {mon_hoc}"
        if phong:
            base_answer += f", kiểm tra lại phòng {phong}"
        base_answer += " và có mặt sớm trước giờ thi."

    if conflicts:
        base_answer += f" Hệ thống phát hiện {conflicts} mục có khả năng trùng lịch theo ngày giờ."
    elif len(entries) >= 2:
        base_answer += " Hiện chưa thấy lịch thi bị trùng giờ trong các dòng đọc được."

    if fallback_answer and len(fallback_answer.strip()) > len(base_answer.strip()) + 40:
        return fallback_answer.strip()
    return base_answer.strip()


def _build_exam_reasoning(entries: List[Dict[str, Any]], fallback_reasoning: str) -> str:
    if fallback_reasoning:
        return fallback_reasoning
    if not entries:
        return ""
    conflicts = _detect_exam_conflicts(entries)
    if conflicts:
        return "Dựa trên danh sách lịch thi đã trích xuất, hệ thống so sánh ngày và giờ để xác định môn thi sớm nhất và kiểm tra các mục có khả năng trùng lịch."
    return "Dựa trên danh sách lịch thi đã trích xuất, hệ thống so sánh ngày và giờ để xác định môn thi sớm nhất và kiểm tra xem có lịch thi bị trùng hay không."


def _build_exam_recommendations(entries: List[Dict[str, Any]], existing: List[str]) -> List[str]:
    if existing:
        return existing[:3]
    if not entries:
        return []
    earliest = entries[0]
    mon_hoc = earliest.get("mon_hoc") or "môn thi đầu tiên"
    phong = earliest.get("phong")
    recs = [f"Ưu tiên ôn tập môn {mon_hoc} trước."]
    if phong:
        recs.append(f"Kiểm tra lại địa điểm thi ({phong}) và có mặt sớm 15-30 phút.")
    else:
        recs.append("Kiểm tra lại giờ thi và địa điểm trước khi đến phòng thi.")
    if _detect_exam_conflicts(entries):
        recs.append("Nếu thấy trùng lịch thật, hãy liên hệ Phòng Đào tạo hoặc cố vấn học tập sớm.")
    else:
        recs.append("Chuẩn bị giấy tờ cần thiết như thẻ sinh viên/CCCD, bút và dụng cụ được phép mang vào phòng thi.")
    return recs[:3]


def _build_exam_highlights(entries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not entries:
        return []
    earliest = entries[0]
    highlights = [
        {"label": "Môn thi đầu tiên", "value": earliest.get("mon_hoc") or "Chưa rõ"},
    ]
    when_parts = [part for part in [earliest.get("gio_thi"), earliest.get("ngay_thi")] if part]
    if when_parts:
        highlights.append({"label": "Thời gian", "value": " • ".join(when_parts)})
    if earliest.get("phong"):
        highlights.append({"label": "Phòng", "value": earliest["phong"]})
    highlights.append(
        {
            "label": "Trùng lịch",
            "value": "Có khả năng trùng" if _detect_exam_conflicts(entries) else "Chưa thấy trùng",
        }
    )
    return highlights[:4]


def _enrich_exam_schedule_result(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    extracted_data = _safe_mapping(result.get("extracted_data"))
    entries = _extract_exam_entries(extracted_data)
    if not entries:
        return result

    explicit_conflict = _extract_exam_conflict_flag(extracted_data)
    conflict_count = _detect_exam_conflicts(entries)
    if explicit_conflict is True and conflict_count == 0:
        conflict_count = 1

    fallback_answer = "" if _needs_regenerated_answer(result.get("answer")) else result.get("answer", "")
    result["answer"] = _build_exam_answer(entries, question, fallback_answer)
    result["reasoning"] = _build_exam_reasoning(entries, result.get("reasoning", ""))
    result["recommendations"] = _build_exam_recommendations(entries, result.get("recommendations", []))
    result["highlights"] = _build_exam_highlights(entries)
    if explicit_conflict is not None and result.get("highlights"):
        result["highlights"][-1]["value"] = "Có khả năng trùng" if explicit_conflict else "Chưa thấy trùng"
        answer_key = _normalize_key(result["answer"])
        if explicit_conflict and "TRUNG LICH" not in answer_key:
            result["answer"] += " Hệ thống đọc được dấu hiệu có khả năng trùng lịch."
        elif explicit_conflict is False and "TRUNG LICH" not in answer_key:
            result["answer"] += " Chưa thấy dấu hiệu trùng lịch trong phần dữ liệu đọc được."

    earliest = entries[0]
    result["structured_summary"] = {
        "earliest_exam": earliest.get("mon_hoc") or "",
        "earliest_time": earliest.get("gio_thi") or "",
        "earliest_date": earliest.get("ngay_thi") or "",
        "room": earliest.get("phong") or "",
        "conflicts": conflict_count,
        "entries_count": len(entries),
    }

    essential_fields = sum(
        1 for field in [earliest.get("mon_hoc"), earliest.get("ngay_thi"), earliest.get("gio_thi"), earliest.get("phong")] if field
    )
    if essential_fields >= 3:
        result["confidence"] = "high"
    elif essential_fields >= 2:
        result["confidence"] = "medium"
    return result


def _enrich_notice_result(result: Dict[str, Any]) -> Dict[str, Any]:
    extracted_data = _safe_mapping(result.get("extracted_data"))
    title = _clean_text(_pick_value(extracted_data, "tieu_de", "title"))
    deadline = _clean_text(_pick_value(extracted_data, "deadline", "han_chot"))
    action = _clean_text(_pick_value(extracted_data, "yeu_cau", "hanh_dong", "viec_can_lam"))
    audience = _clean_text(_pick_value(extracted_data, "doi_tuong", "audience"))
    summary = _clean_text(_pick_value(extracted_data, "noi_dung_chinh", "summary"))

    if not any([title, deadline, action, audience, summary]):
        return result

    if _needs_regenerated_answer(result.get("answer")) or len(_clean_text(result.get("answer"))) < 60:
        parts = []
        if title:
            parts.append(f"Thông báo chính liên quan đến {title}.")
        elif summary:
            parts.append(f"Nội dung chính của thông báo là: {summary}.")
        if deadline:
            parts.append(f"Deadline đáng chú ý là {deadline}.")
        if action:
            parts.append(f"Việc cần làm là {action}.")
        result["answer"] = " ".join(parts).strip() or result.get("answer", "")

    if not result.get("recommendations"):
        recs = []
        if deadline:
            recs.append(f"Ghi chú deadline {deadline} để tránh bỏ sót.")
        if action:
            recs.append(f"Ưu tiên hoàn thành yêu cầu: {action}.")
        if audience:
            recs.append(f"Kiểm tra xem bạn có thuộc đối tượng áp dụng: {audience}.")
        result["recommendations"] = recs[:3]

    result["highlights"] = [
        item for item in [
            {"label": "Tiêu đề", "value": title} if title else None,
            {"label": "Deadline", "value": deadline} if deadline else None,
            {"label": "Đối tượng", "value": audience} if audience else None,
        ] if item
    ]

    strong_fields = sum(1 for value in [title, deadline, action, summary] if value)
    if strong_fields >= 2:
        result["confidence"] = "high"
    elif strong_fields == 1 and result.get("confidence") == "low":
        result["confidence"] = "medium"
    return result


def _flatten_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        items = []
        for item in value:
            item_text = _clean_text(item)
            if item_text:
                items.append(item_text)
        return items
    text = _clean_text(value)
    return [text] if text else []


def _enrich_screenshot_result(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    extracted_data = _safe_mapping(result.get("extracted_data"))
    title = _clean_text(_pick_value(extracted_data, "tieu_de", "title"))
    status = _clean_text(_pick_value(extracted_data, "trang_thai", "status"))
    note = _clean_text(_pick_value(extracted_data, "ghi_chu", "note", "summary"))
    main_values = _flatten_string_list(_pick_value(extracted_data, "chi_so_chinh", "metrics", "key_texts", "main_values"))
    question_key = _normalize_key(question)
    asks_quota = any(token in question_key for token in ["QUOTA", "RPM", "RPD", "TPM", "LIMIT"])

    if not any([title, status, note, main_values]):
        return result

    if asks_quota or _needs_regenerated_answer(result.get("answer")) or len(_clean_text(result.get("answer"))) < 50:
        parts = []
        if title:
            parts.append(f"Ảnh chụp màn hình đang hiển thị mục {title}.")
        if asks_quota:
            if main_values:
                parts.append("Các chỉ số đọc được gồm: " + "; ".join(main_values[:3]) + ".")
            elif note:
                parts.append(note + ".")
        else:
            if note:
                parts.append(note + ".")
            elif main_values:
                parts.append("Thông tin nổi bật nhìn thấy: " + "; ".join(main_values[:3]) + ".")
        if status:
            parts.append(f"Trạng thái hiển thị: {status}.")
        result["answer"] = " ".join(parts).strip() or result.get("answer", "")

    if not result.get("recommendations"):
        recs = []
        if asks_quota:
            recs.append("Đối chiếu các chỉ số đang hiển thị với giới hạn RPM/RPD để biết còn quota hay không.")
            recs.append("Nếu các chỉ số gần chạm ngưỡng, nên giảm số request hoặc chờ sang ngày mới.")
        elif main_values:
            recs.append("Xem lại các chỉ số/chữ chính đang hiển thị để xác nhận thông tin quan trọng nhất.")
        if title:
            recs.append(f"Nếu cần chính xác hơn, hãy hỏi cụ thể hơn theo nội dung trong mục {title}.")
        result["recommendations"] = recs[:3]

    result["highlights"] = [
        item for item in [
            {"label": "Tiêu đề", "value": title} if title else None,
            {"label": "Trạng thái", "value": status} if status else None,
            {"label": "Chỉ số chính", "value": main_values[0]} if main_values else None,
        ] if item
    ]

    signal_count = sum(1 for value in [title, status, note] if value) + (1 if main_values else 0)
    if signal_count >= 2:
        result["confidence"] = "medium" if result.get("confidence") == "low" else result.get("confidence")
    return result


def _enrich_receipt_result(result: Dict[str, Any]) -> Dict[str, Any]:
    extracted_data = _safe_mapping(result.get("extracted_data"))
    ma_sv = _clean_text(_pick_value(extracted_data, "ma_sv", "mssv"))
    ho_ten = _clean_text(_pick_value(extracted_data, "ho_ten", "ten_sv", "ho_ten_sv"))
    so_tien = _clean_text(_pick_value(extracted_data, "so_tien", "tong_tien", "amount"))
    ngay_dong = _clean_text(_pick_value(extracted_data, "ngay_dong", "ngay_thanh_toan", "payment_date"))
    ngan_hang = _clean_text(_pick_value(extracted_data, "ngan_hang", "bank"))
    trang_thai = _clean_text(_pick_value(extracted_data, "trang_thai", "status"))

    if not any([ma_sv, ho_ten, so_tien, ngay_dong, ngan_hang, trang_thai]):
        return result

    if _needs_regenerated_answer(result.get("answer")) or len(_clean_text(result.get("answer"))) < 60:
        parts = ["Biên lai học phí đã được trích xuất với các thông tin chính"]
        details = []
        if ma_sv:
            details.append(f"mã SV {ma_sv}")
        if ho_ten:
            details.append(f"sinh viên {ho_ten}")
        if so_tien:
            details.append(f"số tiền {so_tien}")
        if ngay_dong:
            details.append(f"ngày đóng {ngay_dong}")
        if details:
            parts = [f"Biên lai học phí thể hiện {', '.join(details)}."]
        if trang_thai:
            parts.append(f"Trạng thái đọc được: {trang_thai}.")
        result["answer"] = " ".join(parts).strip()

    if not result.get("recommendations"):
        recs = []
        if so_tien:
            recs.append("Đối chiếu số tiền trên biên lai với thông báo học phí của trường.")
        if ma_sv:
            recs.append("Kiểm tra lại mã sinh viên trên biên lai để tránh nhầm lẫn.")
        recs.append("Lưu lại biên lai hoặc ảnh chụp để đối chiếu khi cần.")
        result["recommendations"] = recs[:3]

    result["highlights"] = [
        item for item in [
            {"label": "Mã SV", "value": ma_sv} if ma_sv else None,
            {"label": "Số tiền", "value": so_tien} if so_tien else None,
            {"label": "Ngày đóng", "value": ngay_dong} if ngay_dong else None,
            {"label": "Trạng thái", "value": trang_thai} if trang_thai else None,
        ] if item
    ]

    strong_fields = sum(1 for value in [ma_sv, so_tien, ngay_dong, trang_thai] if value)
    if strong_fields >= 3:
        result["confidence"] = "high"
    elif strong_fields >= 2 and result.get("confidence") == "low":
        result["confidence"] = "medium"
    return result


def _enrich_grade_report_result(result: Dict[str, Any]) -> Dict[str, Any]:
    extracted_data = _safe_mapping(result.get("extracted_data"))
    gpa = _clean_text(_pick_value(extracted_data, "gpa_uoc_tinh", "gpa", "dtb"))
    hoc_ky = _clean_text(_pick_value(extracted_data, "hoc_ky", "semester"))
    mon_hoc = _pick_value(extracted_data, "mon_hoc", "danh_sach_mon", "subjects")
    subjects = [item for item in mon_hoc if isinstance(item, dict)] if isinstance(mon_hoc, list) else []

    low_subjects = []
    for subject in subjects:
        ten_mon = _clean_text(_pick_value(subject, "ten_mon", "mon_hoc", "mon"))
        diem = _clean_text(_pick_value(subject, "diem", "score"))
        if ten_mon and diem:
            low_subjects.append(f"{ten_mon} ({diem})")
        if len(low_subjects) >= 2:
            break

    if not any([gpa, hoc_ky, subjects]):
        return result

    if _needs_regenerated_answer(result.get("answer")) or len(_clean_text(result.get("answer"))) < 60:
        parts = []
        if hoc_ky:
            parts.append(f"Bảng điểm thuộc {hoc_ky}.")
        if gpa:
            parts.append(f"GPA ước tính đọc được là {gpa}.")
        if low_subjects:
            parts.append(f"Một số môn bạn nên kiểm tra thêm là {', '.join(low_subjects)}.")
        result["answer"] = " ".join(parts).strip() or result.get("answer", "")

    if not result.get("recommendations"):
        recs = []
        if gpa:
            recs.append(f"Theo dõi GPA hiện tại ({gpa}) để đánh giá cơ hội học bổng/cảnh báo học vụ.")
        if low_subjects:
            recs.append(f"Ưu tiên cải thiện các môn: {', '.join(low_subjects)}.")
        recs.append("Đối chiếu lại điểm với bảng điểm gốc trên cổng thông tin nếu cần.")
        result["recommendations"] = recs[:3]

    result["highlights"] = [
        item for item in [
            {"label": "Học kỳ", "value": hoc_ky} if hoc_ky else None,
            {"label": "GPA ước tính", "value": gpa} if gpa else None,
            {"label": "Số môn đọc được", "value": str(len(subjects))} if subjects else None,
        ] if item
    ]

    if gpa and subjects:
        result["confidence"] = "high"
    elif (gpa or subjects) and result.get("confidence") == "low":
        result["confidence"] = "medium"
    return result


def _enhance_image_result(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    result["image_type"] = _normalize_image_type(result.get("image_type"))
    result["extracted_data"] = _safe_mapping(result.get("extracted_data"))
    result["reasoning"] = _clean_text(result.get("reasoning"))
    result["answer"] = _clean_text(result.get("answer"))
    result["recommendations"] = _normalize_recommendations(result.get("recommendations"))
    result["confidence"] = _normalize_confidence(result.get("confidence"))

    image_type_key = _normalize_key(result.get("image_type"))
    question_key = _normalize_key(question)
    has_dashboard_metrics = bool(
        _pick_value(result.get("extracted_data", {}), "chi_so_chinh", "metrics", "key_texts", "main_values")
    )
    if image_type_key == "LICH THI":
        result = _enrich_exam_schedule_result(result, question)
    elif image_type_key == "THONG BAO":
        result = _enrich_notice_result(result)
        if has_dashboard_metrics or any(token in question_key for token in ["QUOTA", "RPM", "RPD", "TPM", "LIMIT"]):
            result = _enrich_screenshot_result(result, question)
        elif _needs_regenerated_answer(result.get("answer")):
            result = _enrich_screenshot_result(result, question)
    elif image_type_key == "SCREENSHOT":
        result = _enrich_screenshot_result(result, question)
    elif image_type_key == "HOA DON HOC PHI":
        result = _enrich_receipt_result(result)
    elif image_type_key == "BANG DIEM":
        result = _enrich_grade_report_result(result)

    if not result.get("answer"):
        result["answer"] = "Đã nhận diện được một phần nội dung, nhưng chưa đủ dữ liệu để trả lời rõ ràng."
    if not result.get("recommendations"):
        result["recommendations"] = ["Thử hỏi rõ hơn vào đúng trọng tâm của tài liệu để Gemini suy luận chính xác hơn."]
    if result["image_type"] != "không xác định" and result["confidence"] == "low":
        has_structure = bool(result.get("extracted_data")) or bool(result.get("highlights"))
        if has_structure and len(result.get("answer", "")) >= 40:
            result["confidence"] = "medium"
    return result


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
        cached   = _vision_cache.get(IMAGE_CACHE_NAMESPACE, img_hash, question)
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
                    response_mime_type="application/json",
                ),
            )

        def _is_temporary_vision_error(err_text: str) -> bool:
            lowered = err_text.lower()
            return any(marker in lowered for marker in [
                "503",
                "unavailable",
                "high demand",
                "currently experiencing high demand",
                "temporarily unavailable",
                "timeout",
                "internal",
            ])

        if _daily_tracker and not _daily_tracker.can_consume("gemini_vision", GEMINI_VISION_RPD_SOFT):
            return _error_response(
                "quota",
                "Đã chạm soft cap Gemini trong ngày để tránh hết quota free. Hãy thử lại vào ngày mới hoặc dùng fallback."
            )

        for attempt in range(4):
            try:
                if _vision_limiter:
                    with _vision_limiter:
                        response = _call()
                else:
                    response = _call()
                break
            except Exception as e:
                err = str(e)
                if attempt < 3 and _is_temporary_vision_error(err):
                    delay = 2 * (2 ** attempt)
                    print(f"[Vision] Gemini tam thoi qua tai, retry sau {delay}s ({attempt + 1}/3)...")
                    time.sleep(delay)
                    continue
                raise

        if _counter:
            _counter.increment("gemini_vision")
        if _daily_tracker:
            _daily_tracker.increment("gemini_vision")

        result               = _enhance_image_result(_parse_image_response(_response_to_text(response)), question)
        result["from_cache"] = False

        # ── Lưu cache ──
        if _vision_cache:
            img_hash = hashlib.md5(image_bytes).hexdigest()
            _vision_cache.set(result, IMAGE_CACHE_NAMESPACE, img_hash, question)

        return result

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            return _error_response(
                "quota",
                "Hết quota Gemini Vision. Chờ 1 phút rồi thử lại (reset lúc 07:00 VN)."
            )
        if "503" in err or "UNAVAILABLE" in err or "high demand" in err.lower():
            return _error_response(
                "unavailable",
                "Gemini Vision dang qua tai tam thoi. App da tu thu lai vai lan nhung chua thanh cong, vui long thu lai sau it phut."
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
            return _media_error_response(
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

        def _is_temporary_file_api_error(err_text: str) -> bool:
            lowered = err_text.lower()
            return any(marker in lowered for marker in [
                "503",
                "unavailable",
                "high demand",
                "currently experiencing high demand",
                "temporarily unavailable",
                "timeout",
                "internal",
            ])

        if _daily_tracker and not _daily_tracker.can_consume("gemini_vision", GEMINI_VISION_RPD_SOFT):
            return _media_error_response(
                "quota",
                "Đã chạm soft cap Gemini trong ngày để tránh hết quota free. Hãy thử lại vào ngày mới hoặc dùng fallback."
            )

        for attempt in range(4):
            try:
                if _vision_limiter:
                    with _vision_limiter:
                        response = _call()
                else:
                    response = _call()
                break
            except Exception as e:
                err = str(e)
                if attempt < 3 and _is_temporary_file_api_error(err):
                    delay = 3 * (2 ** attempt)
                    print(f"[File API] Gemini tam thoi qua tai, retry sau {delay}s ({attempt + 1}/3)...")
                    time.sleep(delay)
                    continue
                raise

        if _counter:
            _counter.increment("gemini_vision")
        if _daily_tracker:
            _daily_tracker.increment("gemini_vision")

        result = _parse_media_response(_response_to_text(response))
        result["from_cache"] = False
        if _vision_cache:
            _vision_cache.set(result, "media", file_hash, question)
        return result

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            return _media_error_response("quota", "Hết quota Gemini. Reset lúc 07:00 VN.")
        if "503" in err or "UNAVAILABLE" in err or "high demand" in err.lower():
            return _media_error_response(
                "unavailable",
                "Gemini File API dang qua tai tam thoi. App da tu thu lai vai lan nhung chua thanh cong, vui long thu lai sau it phut."
            )
        return _media_error_response("api", f"Lỗi File API: {err[:200]}")

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
    extracted_data = _extract_object_field(text, "extracted_data")

    if answer or image_type or extracted_data:
        return {
            "image_type": image_type or "không xác định",
            "extracted_data": extracted_data,
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


def _media_error_response(error_type: str, message: str) -> dict:
    return {
        "content_type": f"lỗi_{error_type}",
        "summary": [],
        "action_items": [],
        "key_moments": [],
        "answer": message,
        "confidence": "low",
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
