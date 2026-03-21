# =============================================================
# multimodal.py - PHÂN TÍCH ẢNH VỚI GEMINI VISION (LEVEL 2)
# =============================================================
# Use cases chính trong bối cảnh SGU:
#   1. Phân tích hóa đơn học phí chụp ảnh
#   2. Đọc bảng điểm chụp từ website
#   3. Phân tích lịch thi chụp ảnh
#   4. Đọc thông báo dán bảng/poster
#
# Tại sao dùng Gemini Vision:
#   - Cùng API key với LLM chính → không cần key riêng
#   - gemini-2.0-flash hỗ trợ multimodal (ảnh + text)
#   - Miễn phí trong giới hạn 1500 req/ngày
#
# Lý do KHÔNG dùng "import google.generativeai as genai" trực tiếp:
#   - google-generativeai==0.8.5 gây conflict với langchain-google-genai==2.1.4
#     (cả hai cùng kéo google-ai-generativelanguage nhưng khác version)
#   - Thay vào đó dùng ChatGoogleGenerativeAI từ langchain-google-genai
#     + truyền ảnh dưới dạng base64 qua HumanMessage content list
#   - Cách này không cần cài thêm package, dùng đúng những gì đã có
#
# Structured Output: Tất cả trả về JSON chuẩn giống rag_chain.py
# =============================================================

import os        # Đọc biến môi trường (API key)
import base64    # Encode ảnh thành base64 để gửi qua API
import json      # Parse JSON response từ LLM
import re        # Regex để xóa markdown từ response
from typing import Dict, Any   # Type hints cho rõ ràng
from dotenv import load_dotenv # Nạp .env file
from PIL import Image          # Xử lý ảnh: validate, resize
import io                      # BytesIO để xử lý ảnh trong bộ nhớ

# Dùng ChatGoogleGenerativeAI từ langchain thay vì google.generativeai trực tiếp
# → tránh conflict package, vẫn dùng được Gemini Vision
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage  # Định dạng message gửi LLM

load_dotenv()  # Nạp các biến từ file .env (GOOGLE_API_KEY, v.v.)

# Lấy API key từ .env, mặc định rỗng nếu chưa cài
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# =============================================================
# SYSTEM PROMPT CHO PHÂN TÍCH ẢNH
# =============================================================
# Lý do viết prompt này:
# - Định hướng Gemini Vision chỉ tập trung vào tài liệu học thuật SGU
# - Yêu cầu trả về JSON để đồng nhất với response của RAG chain
# - Phân biệt các loại tài liệu để extract đúng thông tin
# =============================================================

IMAGE_ANALYSIS_PROMPT = """Bạn là trợ lý phân tích tài liệu của sinh viên Trường Đại học Sài Gòn (SGU).

Hãy phân tích ảnh được cung cấp và trả lời câu hỏi sau: {question}

Hướng dẫn phân tích theo loại tài liệu:
- HÓA ĐƠN HỌC PHÍ: Trích xuất mã SV, tên, số tiền, ngày đóng, ngân hàng
- BẢNG ĐIỂM: Liệt kê tên môn, điểm, số tín chỉ, GPA nếu có
- LỊCH THI: Liệt kê môn thi, ngày giờ, phòng thi, hình thức thi
- THÔNG BÁO: Tóm tắt nội dung chính, deadline quan trọng

YÊU CẦU FORMAT OUTPUT:
Trả về JSON hợp lệ, KHÔNG kèm markdown:
{{
  "image_type": "loại tài liệu (hóa đơn/bảng điểm/lịch thi/thông báo/khác)",
  "extracted_data": {{
    "ghi chú": "dữ liệu được trích xuất theo loại tài liệu"
  }},
  "answer": "Câu trả lời cho câu hỏi của người dùng",
  "important_notes": ["lưu ý 1", "lưu ý 2"],
  "confidence": "high/medium/low"
}}

Nếu ảnh mờ hoặc không đọc được, hãy nói rõ trong "answer"."""


# =============================================================
# HÀM PHÂN TÍCH ẢNH CHÍNH
# =============================================================

def analyze_image(image_bytes: bytes, question: str) -> Dict[str, Any]:
    """
    Phân tích ảnh tài liệu và trả lời câu hỏi của sinh viên.

    Cách hoạt động:
    1. Validate và resize ảnh (nếu quá lớn) để tiết kiệm token
    2. Encode ảnh thành base64
    3. Gửi tới Gemini Vision qua ChatGoogleGenerativeAI (không dùng google.generativeai)
    4. Parse JSON từ response và trả về dict chuẩn

    Args:
        image_bytes: Dữ liệu ảnh dạng bytes (từ st.file_uploader)
        question:    Câu hỏi của sinh viên về nội dung ảnh

    Returns:
        Dict chuẩn: image_type, extracted_data, answer, important_notes, confidence
    """
    # --- Kiểm tra API key trước khi làm gì cả ---
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "Thiếu GOOGLE_API_KEY! Hãy cài đặt trong file .env\n"
            "Lấy key miễn phí tại: https://aistudio.google.com/app/apikey"
        )

    # --- Bước 1: Validate và resize ảnh ---
    try:
        # Mở ảnh bằng PIL để kiểm tra hợp lệ và đọc kích thước
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Resize nếu ảnh quá lớn → tiết kiệm token (Token Management)
        # Gemini vẫn đọc được ảnh 1024x1024, không cần gửi ảnh 4K
        max_size = (1024, 1024)
        if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
            # thumbnail() giữ tỉ lệ aspect ratio, không méo ảnh
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            print(f"[Multimodal] Resize ảnh xuống {pil_image.size} (Token Management)")

        # Xác định format ảnh để encode đúng MIME type
        img_format = pil_image.format if pil_image.format else "JPEG"
        # Chỉ giữ các format Gemini hỗ trợ, còn lại convert sang JPEG
        if img_format.upper() not in ["JPEG", "PNG", "WEBP", "GIF"]:
            img_format = "JPEG"

        # Ghi ảnh đã resize vào buffer bytes để encode base64
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format=img_format)
        processed_bytes = img_buffer.getvalue()  # Lấy bytes đã xử lý

    except Exception as e:
        # Trả về error có cấu trúc JSON thay vì raise exception → không crash app
        return {
            "image_type": "lỗi",
            "extracted_data": {},
            "answer": f"Không thể đọc ảnh: {str(e)}. Vui lòng thử lại với ảnh JPG/PNG/WEBP.",
            "important_notes": ["Ảnh cần ở định dạng JPG, PNG hoặc WEBP"],
            "confidence": "low",
        }

    # --- Bước 2: Khởi tạo Gemini Vision qua LangChain ---
    # Dùng ChatGoogleGenerativeAI thay vì google.generativeai.GenerativeModel
    # Lý do: không cần cài google-generativeai riêng → tránh conflict package
    llm_vision = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",   # Model hỗ trợ multimodal, miễn phí
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,            # Thấp = nhất quán, ít sáng tạo
        max_output_tokens=1024,     # Giới hạn output (Token Management)
    )

    # --- Bước 3: Tạo message multimodal (text + image) ---
    try:
        # Encode ảnh thành base64 string để nhúng vào message
        b64_image = base64.b64encode(processed_bytes).decode("utf-8")

        # Format prompt với câu hỏi của người dùng
        prompt_text = IMAGE_ANALYSIS_PROMPT.format(question=question)

        # HumanMessage với content dạng list = multimodal message
        # Phần tử 1: text prompt
        # Phần tử 2: ảnh base64 với MIME type
        message = HumanMessage(
            content=[
                # Phần text: prompt hướng dẫn phân tích
                {
                    "type": "text",
                    "text": prompt_text,
                },
                # Phần ảnh: base64 encoded image
                {
                    "type": "image_url",
                    "image_url": {
                        # Data URL format: "data:<mime>base64,<data>"
                        "url": f"data:image/{img_format.lower()};base64,{b64_image}",
                    },
                },
            ]
        )

        # Gọi Gemini Vision API qua LangChain
        response = llm_vision.invoke([message])
        raw_text = response.content  # Lấy text content từ response

    except Exception as e:
        error_msg = str(e)

        # Xử lý riêng lỗi hết quota (HTTP 429)
        if "quota" in error_msg.lower() or "429" in error_msg:
            return {
                "image_type": "lỗi quota",
                "extracted_data": {},
                "answer": (
                    "Đã hết quota Gemini hôm nay (1500 req/ngày). "
                    "Vui lòng thử lại vào ngày mai hoặc dùng tab Chat để hỏi bằng text."
                ),
                "important_notes": ["Quota Gemini reset lúc 07:00 sáng giờ VN (00:00 UTC)"],
                "confidence": "low",
            }

        # Các lỗi API khác (sai key, network, v.v.)
        return {
            "image_type": "lỗi API",
            "extracted_data": {},
            "answer": f"Lỗi khi gọi Gemini Vision: {error_msg[:200]}",
            "important_notes": ["Kiểm tra GOOGLE_API_KEY trong file .env"],
            "confidence": "low",
        }

    # --- Bước 4: Parse JSON từ response của Gemini ---
    return _parse_vision_response(raw_text)


def _parse_vision_response(raw_text: str) -> dict:
    """
    Parse JSON từ response text của Gemini Vision.

    LLM đôi khi trả về JSON kèm markdown (```json...```) hoặc text thừa.
    Hàm này dọn sạch và parse về dict chuẩn.

    Args:
        raw_text: Response text thô từ Gemini

    Returns:
        Dict chuẩn với đầy đủ keys cần thiết
    """
    # Xóa markdown code block nếu có: ```json ... ``` hoặc ``` ... ```
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)  # Xóa mở đầu ```json
    text = re.sub(r"\s*```$", "", text)            # Xóa kết thúc ```

    try:
        # Parse JSON string thành Python dict
        data = json.loads(text)

        # Trả về dict với đầy đủ keys, dùng .get() để tránh KeyError
        return {
            "image_type":      data.get("image_type", "không xác định"),
            "extracted_data":  data.get("extracted_data", {}),
            "answer":          data.get("answer", "Không thể phân tích ảnh này."),
            "important_notes": data.get("important_notes", []),
            "confidence":      data.get("confidence", "low"),
        }

    except json.JSONDecodeError:
        # Fallback khi LLM không trả về JSON hợp lệ
        # Vẫn trả về dict có cấu trúc, không crash app
        return {
            "image_type":      "không xác định",
            "extracted_data":  {},
            "answer":          raw_text,  # Giữ nguyên text gốc để người dùng đọc
            "important_notes": ["Kết quả không phải JSON chuẩn (lỗi parse)"],
            "confidence":      "low",
        }


def get_image_dimensions(image_bytes: bytes) -> tuple:
    """
    Lấy kích thước ảnh (width, height) để hiển thị trên UI.

    Args:
        image_bytes: Dữ liệu ảnh dạng bytes

    Returns:
        Tuple (width, height), hoặc (0, 0) nếu lỗi
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))  # Mở ảnh
        return img.size  # PIL trả về (width, height)
    except Exception:
        return (0, 0)  # Trả về (0,0) thay vì crash nếu ảnh lỗi


def is_valid_image(image_bytes: bytes) -> bool:
    """
    Kiểm tra bytes có phải là ảnh hợp lệ không.
    Dùng để validate trước khi gọi API.

    Args:
        image_bytes: Dữ liệu cần kiểm tra

    Returns:
        True nếu là ảnh hợp lệ, False nếu không phải
    """
    try:
        Image.open(io.BytesIO(image_bytes))  # Thử mở, nếu thành công là ảnh hợp lệ
        return True
    except Exception:
        return False  # PIL không đọc được → không phải ảnh hợp lệ


# =============================================================
# TEST NHANH - chạy: python src/multimodal.py <đường_dẫn_ảnh>
# =============================================================
if __name__ == "__main__":
    import sys

    # Hướng dẫn sử dụng khi chạy trực tiếp
    if len(sys.argv) < 2:
        print("Cách dùng: python src/multimodal.py <đường_dẫn_ảnh>")
        print("Ví dụ:     python src/multimodal.py test_invoice.jpg")
        sys.exit(1)

    image_path = sys.argv[1]  # Lấy đường dẫn ảnh từ argument

    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        print(f"[X] Không tìm thấy file: {image_path}")
        sys.exit(1)

    # Đọc file ảnh thành bytes
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Câu hỏi mặc định khi test
    question = "Đây là loại tài liệu gì? Hãy trích xuất thông tin quan trọng."

    print(f"\n📸 Phân tích ảnh: {image_path}")
    print(f"❓ Câu hỏi: {question}\n")

    # Gọi hàm phân tích và in kết quả JSON
    result = analyze_image(img_bytes, question)
    print("📋 KẾT QUẢ:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
