# =============================================================
# src/llm_router.py - TỰ ĐỘNG CHUYỂN ĐỔI LLM KHI HẾT QUOTA
# =============================================================
# Thứ tự fallback:
#   1. Gemini 2.0 Flash  → Free 1500 req/ngày, tốt nhất cho tiếng Việt
#   2. Groq llama-3.1-8b → Free 14400 req/ngày, rất nhanh
#   3. Ollama qwen2.5:3b → Local không giới hạn, cần cài sẵn
# =============================================================

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

GEMINI_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GROQ_MODEL   = "llama-3.1-8b-instant"
OLLAMA_MODEL = "qwen2.5:3b"

_current_provider = "Chưa khởi tạo"


def get_current_provider() -> str:
    return _current_provider


def _try_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise ValueError("Thiếu GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        max_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "960")),
        model_kwargs={"response_mime_type": "application/json"},
    )


def _try_groq():
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise ValueError("Thiếu GROQ_API_KEY")
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1024,
    )


def _try_ollama():
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=1024,
    )
    llm.invoke("Hi")  # Test kết nối
    return llm


def get_llm(force_provider: str = None, allow_fallback: bool = True):
    """
    Trả về (llm_object, provider_name).
    Tự động fallback Gemini → Groq → Ollama khi hết quota.
    """
    global _current_provider

    providers = [
        (f"Gemini {GEMINI_MODEL}",  _try_gemini),
        ("Groq llama-3.1-8b", _try_groq),
        ("Ollama qwen2.5:3b", _try_ollama),
    ]

    if force_provider:
        force_map = {
            "gemini": (f"Gemini {GEMINI_MODEL}",  _try_gemini),
            "groq":   ("Groq llama-3.1-8b", _try_groq),
            "ollama": ("Ollama qwen2.5:3b", _try_ollama),
        }
        if force_provider in force_map:
            forced   = force_map[force_provider]
            providers = [forced] if not allow_fallback else [forced] + [p for p in providers if p[0] != forced[0]]

    errors = []
    for name, try_func in providers:
        try:
            print(f"[LLM Router] Thử {name}...")
            llm = try_func()
            _current_provider = name
            print(f"[LLM Router] ✅ Dùng: {name}")
            return llm, name
        except Exception as e:
            errors.append(f"{name}: {str(e)[:100]}")
            print(f"[LLM Router] ❌ {name}: {str(e)[:80]}")

    raise RuntimeError(
        "Không khởi tạo được LLM nào!\n" +
        "\n".join(f"  - {e}" for e in errors) +
        "\n\nGiải pháp:\n"
        "  1. Kiểm tra GOOGLE_API_KEY trong .env\n"
        "  2. Kiểm tra GROQ_API_KEY trong .env\n"
        "  3. Đảm bảo Ollama đang chạy: ollama serve"
    )


def get_llm_info() -> dict:
    return {
        f"Gemini {GEMINI_MODEL}": {
            "quota": "5 RPM / 20 req-ngày (soft cap app: 4 RPM / 18 req-ngày)",
            "speed": "Nhanh",
            "cost":  "Miễn phí",
        },
        "Groq llama-3.1-8b": {
            "quota": "14,400 req/ngày",
            "speed": "Rất nhanh (~300 tok/s)",
            "cost":  "Miễn phí",
        },
        "Ollama qwen2.5:3b": {
            "quota": "Không giới hạn",
            "speed": "Phụ thuộc CPU/GPU",
            "cost":  "Miễn phí (local)",
        },
    }
