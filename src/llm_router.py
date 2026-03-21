# =============================================================
# llm_router.py - TỰ ĐỘNG CHUYỂN ĐỔI LLM KHI HẾT QUOTA
# =============================================================
# Chiến lược fallback (Token Management):
#   1. Gemini 2.0 Flash     → Miễn phí 1500 req/ngày, nhanh, thông minh
#   2. Groq llama-3.1-8b    → Miễn phí 14400 req/ngày, rất nhanh
#   3. Ollama qwen2.5:3b    → Local, không giới hạn, chạy offline
#
# Lý do thiết kế router riêng:
# - Tách biệt logic chọn LLM khỏi logic RAG (Single Responsibility)
# - Dễ thêm/bớt provider mà không sửa rag_chain.py
# - Hiển thị provider đang dùng lên UI (yêu cầu báo cáo Token Management)
# =============================================================

import os
from dotenv import load_dotenv
from typing import Tuple

load_dotenv()

# =============================================================
# CẤU HÌNH CÁC PROVIDER
# =============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Tên model cho từng provider
GEMINI_MODEL = "gemini-2.0-flash"          # Miễn phí, quota 1500/ngày
GROQ_MODEL   = "llama-3.3-70b-versatile"   # Miễn phí, quota 14400/ngày
OLLAMA_MODEL = "qwen2.5:3b"                # Local, cần cài ollama pull qwen2.5:3b

# Biến toàn cục lưu provider đang dùng (dùng cho UI)
_current_provider = "Chưa khởi tạo"


def get_current_provider() -> str:
    """Trả về tên provider LLM đang được sử dụng (hiển thị lên UI)."""
    return _current_provider


def _try_gemini():
    """
    Thử khởi tạo Gemini 2.0 Flash — LLM ưu tiên số 1.

    Lý do Gemini là lựa chọn tốt nhất cho RAG:
    - Hiểu tiếng Việt tốt hơn llama
    - Tuân thủ JSON format chính xác hơn
    - Hỗ trợ instruction following tốt hơn model nhỏ

    Token Management:
    - temperature=0.1: nhất quán, ít "sáng tạo" → phù hợp RAG
    - max_output_tokens=1024: giới hạn output tiết kiệm quota

    Lưu ý: KHÔNG gọi llm.invoke() để test vì tốn 1 request quota.
    Chỉ kiểm tra API key hợp lệ là đủ — nếu key sai sẽ fail khi query thật.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Thiếu GOOGLE_API_KEY")

    # Tạo LLM object — không invoke test để tiết kiệm quota
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        max_output_tokens=1024,
    )
    return llm


def _try_groq():
    """
    Thử khởi tạo Groq llama-3.1-8b-instant — backup khi Gemini hết quota.

    Lưu ý quan trọng:
    - llama-3.1-8b là model NHỎ, đôi khi không tuân thủ JSON format
    - Chỉ dùng khi Gemini thật sự hết quota (lỗi 429)
    - Nếu Groq trả về kết quả sai format → rag_chain.py có fallback xử lý
    """
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise ValueError("Thiếu GROQ_API_KEY")

    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1024,
    )
    # KHÔNG invoke test để tránh tốn quota và tránh false-negative
    return llm


def _try_ollama():
    """
    Thử khởi tạo Ollama với model qwen2.5:3b chạy local.
    
    Lưu ý: Cần cài đặt trước:
      1. Tải Ollama: https://ollama.com/download
      2. Chạy: ollama pull qwen2.5:3b
      3. Đảm bảo Ollama đang chạy (ollama serve)
    
    qwen2.5:3b:
    - Kích thước: ~2GB RAM
    - Tốt với tiếng Việt nhờ training đa ngôn ngữ
    - Không cần internet sau khi pull
    """
    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=1024,  # Tương đương max_tokens cho Ollama
    )
    llm.invoke("Hi")
    return llm


def get_llm(force_provider: str = None):
    """
    Hàm chính: Trả về LLM hoạt động đầu tiên theo thứ tự ưu tiên.
    
    Args:
        force_provider: Ép buộc dùng provider cụ thể ('gemini'/'groq'/'ollama')
                       Dùng khi người dùng chọn thủ công trên UI
    
    Returns:
        Tuple(llm_object, provider_name_string)
    
    Logic fallback:
        Thử Gemini → lỗi (hết quota/sai key) → thử Groq → lỗi → dùng Ollama
    """
    global _current_provider
    
    # Danh sách provider theo thứ tự ưu tiên
    providers = [
        ("Gemini 2.0 Flash", _try_gemini),
        ("Groq llama-3.1-8b", _try_groq),
        ("Ollama qwen2.5:3b", _try_ollama),
    ]
    
    # Nếu force một provider cụ thể, đổi thứ tự
    if force_provider:
        force_map = {
            "gemini": ("Gemini 2.0 Flash", _try_gemini),
            "groq":   ("Groq llama-3.1-8b", _try_groq),
            "ollama": ("Ollama qwen2.5:3b", _try_ollama),
        }
        if force_provider in force_map:
            # Đặt provider được chọn lên đầu
            forced = force_map[force_provider]
            providers = [forced] + [p for p in providers if p[0] != forced[0]]
    
    # Thử từng provider theo thứ tự
    errors = []
    for provider_name, try_func in providers:
        try:
            print(f"[LLM Router] Đang thử {provider_name}...")
            llm = try_func()
            _current_provider = provider_name  # Cập nhật provider hiện tại
            print(f"[LLM Router] ✅ Dùng: {provider_name}")
            return llm, provider_name
        except Exception as e:
            error_msg = f"{provider_name}: {str(e)[:100]}"
            errors.append(error_msg)
            print(f"[LLM Router] ❌ {error_msg}")
    
    # Tất cả đều thất bại
    raise RuntimeError(
        "Không thể khởi tạo bất kỳ LLM nào!\n"
        "Lỗi chi tiết:\n" + "\n".join(f"  - {e}" for e in errors) + "\n\n"
        "Giải pháp:\n"
        "  1. Kiểm tra GOOGLE_API_KEY trong .env\n"
        "  2. Kiểm tra GROQ_API_KEY trong .env\n"
        "  3. Đảm bảo Ollama đang chạy: ollama serve"
    )


def get_llm_info() -> dict:
    """
    Trả về thông tin quota/giới hạn của từng provider.
    Dùng để hiển thị hướng dẫn cho người dùng trên UI.
    """
    return {
        "Gemini 2.0 Flash": {
            "quota": "1,500 req/ngày",
            "speed": "Nhanh",
            "quality": "Cao",
            "cost": "Miễn phí",
            "note": "Cần GOOGLE_API_KEY",
        },
        "Groq llama-3.1-8b": {
            "quota": "14,400 req/ngày",
            "speed": "Rất nhanh (~300 tok/s)",
            "quality": "Trung bình",
            "cost": "Miễn phí",
            "note": "Cần GROQ_API_KEY",
        },
        "Ollama qwen2.5:3b": {
            "quota": "Không giới hạn",
            "speed": "Phụ thuộc CPU/GPU",
            "quality": "Trung bình",
            "cost": "Miễn phí (local)",
            "note": "Cần cài Ollama + pull model",
        },
    }
