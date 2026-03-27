# =============================================================
# quota_guard.py - CHỐNG HẾT QUOTA & 429 CHO GEMINI FREE
# =============================================================
# Đặt file này ở thư mục gốc project (cùng cấp app.py)
#
# 3 kỹ thuật chống 429:
#   1. RateLimiter    : Không bao giờ vượt quá RPM thực tế của Gemini Free
#   2. Exponential Backoff: Khi bị 429, chờ 2→4→8 giây rồi tự thử lại
#   3. ResponseCache  : Câu hỏi lặp lại → trả kết quả cũ (0 token tốn)
# =============================================================

import time
import hashlib
import threading
import functools
from collections import deque
from typing import Optional, Any, Callable


# =============================================================
# 1. TOKEN BUCKET RATE LIMITER
# =============================================================
class RateLimiter:
    """
    Giới hạn số request trong cửa sổ 60 giây.

    Gemini Free thực tế:
      - LLM (Flash):  15 RPM  → dùng 14 để an toàn
      - Embedding:    100 RPM → dùng 90 để an toàn
      - Vision:       15 RPM  → dùng 14 (dùng chung quota LLM)

    Khi đã đủ RPM trong 60 giây gần nhất → tự động chờ
    đến khi có slot trống, không cần người dùng làm gì.
    """

    def __init__(self, rpm: int = 14, name: str = "Gemini"):
        self.rpm = rpm
        self.name = name
        self.window = 60.0
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def _wait_if_needed(self):
        with self._lock:
            now = time.time()
            cutoff = now - self.window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.rpm:
                oldest = self._timestamps[0]
                wait_time = (oldest + self.window) - now + 0.2
                if wait_time > 0:
                    print(f"[RateLimiter] {self.name}: đạt {self.rpm} RPM, chờ {wait_time:.1f}s...")
                    time.sleep(wait_time)

            self._timestamps.append(time.time())

    def __enter__(self):
        self._wait_if_needed()
        return self

    def __exit__(self, *args):
        pass

    @property
    def requests_this_minute(self) -> int:
        now = time.time()
        return sum(1 for t in self._timestamps if t > now - self.window)


# Singletons — dùng chung toàn app
_llm_limiter    = RateLimiter(rpm=14, name="Gemini LLM")
_embed_limiter  = RateLimiter(rpm=90, name="Gemini Embed")
_vision_limiter = RateLimiter(rpm=14, name="Gemini Vision")


def get_llm_limiter()    -> RateLimiter: return _llm_limiter
def get_embed_limiter()  -> RateLimiter: return _embed_limiter
def get_vision_limiter() -> RateLimiter: return _vision_limiter


# =============================================================
# 2. EXPONENTIAL BACKOFF
# =============================================================
def with_retry(max_retries: int = 3, base_delay: float = 2.0):
    """
    Decorator retry khi gặp 429 với exponential backoff.

    Lần 1 → chờ 2s → lần 2 → chờ 4s → lần 3 → chờ 8s → raise

    Chỉ retry lỗi 429/quota. Lỗi khác (key sai, network...) raise ngay.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e)
                    is_429 = any(kw in err_str for kw in [
                        "429", "quota", "ResourceExhausted",
                        "RESOURCE_EXHAUSTED", "rate limit", "too many requests",
                    ])
                    if not is_429:
                        raise  # Lỗi không phải quota → raise ngay

                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # 2, 4, 8 giây
                        print(f"[Retry] 429 lần {attempt + 1}/{max_retries}, chờ {delay:.0f}s...")
                        time.sleep(delay)

            raise last_error
        return wrapper
    return decorator


# =============================================================
# 3. RESPONSE CACHE
# =============================================================
class SimpleCache:
    """
    Cache in-memory theo hash của input. FIFO eviction khi đầy.

    RAG cache   : 200 entries (câu hỏi văn bản)
    Vision cache:  50 entries (ảnh nặng hơn)
    """

    def __init__(self, max_size: int = 200):
        self._cache: dict = {}
        self._order: deque = deque()
        self._max_size = max_size
        self._lock = threading.Lock()

    def _key(self, *args) -> str:
        combined = "|".join(str(a) for a in args)
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, *args) -> Optional[Any]:
        key = self._key(*args)
        with self._lock:
            return self._cache.get(key)

    def set(self, value: Any, *args):
        key = self._key(*args)
        with self._lock:
            if key not in self._cache and len(self._cache) >= self._max_size:
                oldest = self._order.popleft()
                self._cache.pop(oldest, None)
            self._cache[key] = value
            if key not in self._order:
                self._order.append(key)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


_rag_cache    = SimpleCache(max_size=200)
_vision_cache = SimpleCache(max_size=50)


def get_rag_cache()    -> SimpleCache: return _rag_cache
def get_vision_cache() -> SimpleCache: return _vision_cache


# =============================================================
# 4. REQUEST COUNTER (hiển thị trên UI sidebar)
# =============================================================
class RequestCounter:
    """Đếm số request đã dùng trong session để hiển thị lên UI."""

    QUOTAS = {
        "gemini_llm":    1500,
        "gemini_embed":  0,     # 0 = không hiển thị progress
        "gemini_vision": 1500,  # Dùng chung quota LLM
        "groq":          14400,
        "cache_hits":    0,
    }

    def __init__(self):
        self._counts: dict = {k: 0 for k in self.QUOTAS}
        self._lock = threading.Lock()

    def increment(self, key: str):
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1

    def get(self, key: str) -> int:
        return self._counts.get(key, 0)

    def get_all(self) -> dict:
        return dict(self._counts)


_counter = RequestCounter()


def get_counter() -> RequestCounter:
    return _counter