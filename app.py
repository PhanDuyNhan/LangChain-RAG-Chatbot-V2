# =============================================================
# app.py - STREAMLIT UI (2 TAB - MỨC 1 + MỨC 2)
# =============================================================
# Tab 1: Chat tài liệu   — RAG Pipeline (Mức 1)
# Tab 2: Đa phương thức  — Visual Reasoning + File API (Mức 2)
#
# Tối ưu quota hiển thị trên sidebar:
#   - Request counter theo từng provider
#   - Cache hit counter
#   - Rate limiter status
# =============================================================

import warnings
import logging

# ── Suppress torch.classes warning (từ chromadb dependency) ──
# Lỗi này vô hại nhưng spam terminal; filter để giữ log sạch
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("torch").setLevel(logging.ERROR)

# ── Suppress FutureWarning từ bất kỳ package cũ nào ──
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import os
import sys
import re
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(
    page_title="SGU Chatbot - Hỗ trợ Sinh viên",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ════════════════════════════════════════
   CHATGPT-LIKE LAYOUT
   ════════════════════════════════════════ */

/* Ẩn Streamlit chrome */
[data-testid="stHeader"] { display: none !important; }
.stApp > header         { display: none !important; }
footer                  { display: none !important; }
#MainMenu               { display: none !important; }

/* Block container — sát trên, không bottom padding thừa */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.2rem !important;
    max-width: 860px;
}

/* ── Tabs: compact, sát top ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    padding: 6px 20px;
    font-size: 0.83rem;
    font-weight: 500;
    border-radius: 0;
}
.stTabs [aria-selected="true"] { font-weight: 700; }

/* ── Chat container: stVerticalBlock > stVerticalBlockBorderWrapper là cấu trúc của st.container() ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    height: calc(100vh - 200px) !important;
    min-height: 200px;
    overflow-y: auto !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
/* Override cho sidebar và columns — không apply height */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpanderDetails"] [data-testid="stVerticalBlockBorderWrapper"] {
    height: auto !important;
    overflow-y: visible !important;
    min-height: unset !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] { padding: 6px 0; }

/* ── Chat input bar ── */
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    font-size: 0.93rem;
}

/* ── Welcome screen: sample question cards ── */
.sample-card button {
    text-align: left !important;
    padding: 10px 14px !important;
    font-size: 0.82rem !important;
    line-height: 1.4 !important;
    white-space: normal !important;
    height: auto !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.04) !important;
    transition: background 0.15s, border-color 0.15s !important;
}
.sample-card button:hover {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.3) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    min-width: 250px !important;
    max-width: 280px !important;
}

/* ── Badges ── */
.provider-badge {
    background: #28a745; color: #fff;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.76em; font-weight: 700;
}
.cache-badge {
    background: #17a2b8; color: #fff;
    padding: 3px 8px; border-radius: 20px;
    font-size: 0.74em; font-weight: 700;
}

/* ── Reasoning box ── */
.reasoning-box {
    background: #fff3cd; border-left: 4px solid #ffc107;
    padding: 10px 12px; border-radius: 5px; margin: 8px 0;
    color: #2b2111; font-size: 0.88em;
}
.reasoning-box b { color: #2b2111; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# CACHE COMPONENTS
# =============================================================
@st.cache_resource(show_spinner="🔄 Đang khởi tạo RAG Chain...")
def load_rag_chain():
    from rag_chain import RAGChain
    return RAGChain()


def check_db_ready() -> bool:
    from ingest import check_db_exists
    return check_db_exists()


# =============================================================
# SIDEBAR
# =============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎓 SGU Chatbot")
        st.markdown("*Hỗ trợ Sinh viên ĐH Sài Gòn*")
        st.divider()

        # ── Trạng thái hệ thống ──
        st.markdown("### 📊 Trạng thái hệ thống")
        if check_db_ready():
            st.success("✅ ChromaDB: Sẵn sàng")
        else:
            st.error("❌ ChromaDB: Chưa có dữ liệu")
            st.info("Chạy: `python src/ingest.py`")

        from dotenv import load_dotenv
        load_dotenv()
        gemini_key = os.getenv("GOOGLE_API_KEY", "")
        groq_key   = os.getenv("GROQ_API_KEY", "")

        if gemini_key and gemini_key != "your_gemini_api_key_here":
            st.success("✅ Gemini API Key: Có")
        else:
            st.warning("⚠️ Gemini API Key: Chưa cài")

        if groq_key and groq_key != "your_groq_api_key_here":
            st.success("✅ Groq API Key: Có")
        else:
            st.warning("⚠️ Groq API Key: Chưa cài")

        st.divider()

        # ── Token Management ──
        st.markdown("### 🔢 Token Management")
        from llm_router import get_current_provider, get_llm_info
        current = get_current_provider()
        if current != "Chưa khởi tạo":
            st.markdown(f'<span class="provider-badge">🤖 {current}</span>', unsafe_allow_html=True)
            st.markdown("")

        with st.expander("📋 Quota các Provider"):
            for provider, details in get_llm_info().items():
                st.markdown(f"**{provider}**")
                st.markdown(f"- Quota: {details['quota']}")
                st.markdown(f"- Tốc độ: {details['speed']}")
                st.markdown(f"- Chi phí: {details['cost']}")
                st.divider()

        st.divider()

        # ── Quota Usage Dashboard ──
        st.markdown("### 📈 Quota Usage (session này)")
        try:
            from quota_guard import (
                get_counter,
                get_llm_limiter,
                get_daily_tracker,
                GEMINI_LLM_RPD_SOFT,
                GEMINI_EMBED_RPD_SOFT,
                GEMINI_VISION_RPD_SOFT,
                GEMINI_TEXT_MODEL,
            )
            counter = get_counter()
            counts  = counter.get_all()
            daily   = get_daily_tracker()

            # Gemini LLM
            used_llm  = counts.get("gemini_llm", 0)
            pct_llm   = min(100, int(used_llm / max(1, GEMINI_LLM_RPD_SOFT) * 100))
            bar_color = "🟢" if pct_llm < 50 else ("🟡" if pct_llm < 80 else "🔴")
            st.markdown(f"{bar_color} **Gemini {GEMINI_TEXT_MODEL}**: {used_llm} / {GEMINI_LLM_RPD_SOFT} req/ngày (session)")
            st.progress(pct_llm / 100)
            st.caption(f"Hôm nay đã dùng toàn app: {daily.get('gemini_llm')} / {GEMINI_LLM_RPD_SOFT}")

            # Embedding
            used_embed = counts.get("gemini_embed", 0)
            pct_embed  = min(100, int(used_embed / max(1, GEMINI_EMBED_RPD_SOFT) * 100))
            st.markdown(f"🟢 **Gemini Embedding**: {used_embed} / {GEMINI_EMBED_RPD_SOFT} req/ngày (session)")
            st.progress(pct_embed / 100)

            # Groq
            used_groq = counts.get("groq", 0)
            pct_groq  = min(100, int(used_groq / 14400 * 100))
            st.markdown(f"🟢 **Groq**: {used_groq} / 14400 req/ngày")
            st.progress(pct_groq / 100)

            # Vision
            used_vis  = counts.get("gemini_vision", 0)
            pct_vis   = min(100, int(used_vis / max(1, GEMINI_VISION_RPD_SOFT) * 100))
            st.markdown(f"🟢 **Gemini Vision/File API**: {used_vis} / {GEMINI_VISION_RPD_SOFT} req/ngày (session)")
            st.progress(pct_vis / 100)

            # Cache hits
            hits = counts.get("cache_hits", 0)
            if hits > 0:
                st.success(f"⚡ Cache tiết kiệm: **{hits} request**")

            # RPM hiện tại
            limiter   = get_llm_limiter()
            rpm_now   = limiter.requests_this_minute
            rpm_limit = limiter.rpm
            rpm_color = "🟢" if rpm_now < max(1, rpm_limit - 2) else ("🟡" if rpm_now < rpm_limit else "🔴")
            st.caption(f"{rpm_color} Gemini RPM: {rpm_now}/{rpm_limit} trong 60 giây gần nhất")

        except ImportError:
            st.caption("quota_guard.py chưa được cài")

        # ── Nút clear cache ──
        st.divider()
        if st.button("🗑️ Xóa Response Cache", use_container_width=True):
            try:
                from quota_guard import get_rag_cache, get_vision_cache
                get_rag_cache().clear()
                get_vision_cache().clear()
                st.success("Đã xóa cache!")
            except ImportError:
                pass


# =============================================================
# TAB 1: CHAT TÀI LIỆU (RAG - Mức 1)
# =============================================================
def render_tab_chat():
    if not check_db_ready():
        st.error("⚠️ Chưa có dữ liệu! Hãy chạy: `python src/ingest.py`")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    samples = [
        "Học phí CNTT năm 2025?",        "Điều kiện học bổng khuyến khích?",
        "Cách đóng học phí online?",      "Thủ tục xin hoãn thi?",
        "Điều kiện tốt nghiệp?",          "Số điện thoại phòng Đào tạo?",
    ]
    has_history = bool(st.session_state.chat_history)

    # ══════════════════════════════════════════════
    # Vùng chat — fill toàn bộ height còn lại
    # ══════════════════════════════════════════════
    # JS: tìm đúng chat container (parent=stVerticalBlock), set height động, auto-scroll
    import streamlit.components.v1 as components
    components.html(f"""
    <script>
    (function() {{
        const run = () => {{
            const doc = window.parent.document;
            const all = [...doc.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]')];

            // Tìm đúng chat container: parent phải là stVerticalBlock, không nằm trong sidebar/column
            const box = all.find(el =>
                el.parentElement?.getAttribute('data-testid') === 'stVerticalBlock' &&
                !el.closest('[data-testid="stSidebar"]') &&
                !el.closest('[data-testid="stColumn"]') &&
                !el.closest('[data-testid="stExpanderDetails"]')
            );
            if (!box) return;

            // Tính height còn lại: viewport - vị trí top của box - chiều cao chat input - margin
            const chatInput = doc.querySelector('[data-testid="stChatInput"]');
            const inputH = chatInput ? chatInput.offsetHeight + 8 : 72;
            const boxTop = box.getBoundingClientRect().top;
            const availH = Math.max(200, window.parent.innerHeight - boxTop - inputH - 4);
            box.style.height = availH + 'px';
            box.style.overflowY = 'auto';
            box.style.border = 'none';
            box.style.boxShadow = 'none';
            box.style.borderRadius = '0';

            // Auto-scroll xuống cuối nếu đang có tin nhắn
            if ({str(has_history).lower()}) {{
                box.scrollTop = box.scrollHeight;
            }}
        }};
        run();
        setTimeout(run, 300);
    }})();
    </script>
    """, height=0)

    with st.container(height=900, border=True):  # border=True để CSS target được element

        if not has_history:
            # ── WELCOME SCREEN ──
            st.markdown("""
            <div style="text-align:center; padding: 48px 0 28px;">
                <div style="font-size:2.8rem; line-height:1; margin-bottom:10px;">🎓</div>
                <h2 style="font-size:1.5rem; font-weight:700; margin:0 0 6px;
                           letter-spacing:-0.01em;">SGU Chatbot</h2>
                <p style="color:rgba(255,255,255,0.45); font-size:0.87rem; margin:0;">
                    Hỏi bất cứ điều gì về Trường Đại học Sài Gòn
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Sample questions — 2 hàng × 3
            for row_start in (0, 3):
                cols = st.columns(3)
                for j, sample in enumerate(samples[row_start:row_start + 3]):
                    if cols[j].button(sample, key=f"sample_{row_start + j}",
                                      use_container_width=True):
                        st.session_state.pending_question = sample

        else:
            # ── CHAT MESSAGES ──
            # Nút "New chat" nhỏ ở góc phải trong vùng messages
            _, clr = st.columns([11, 1])
            with clr:
                if st.button("🗑️", help="Xóa lịch sử", key="clear_history"):
                    st.session_state.chat_history = []
                    st.rerun()

            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    _render_rag_response(chat["response"])


    # ══════════════════════════════════════════════
    # Input — Streamlit render sticky bottom tự động
    # ══════════════════════════════════════════════
    question = st.chat_input("Hỏi về học phí, học bổng, thủ tục...")

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question and question.strip():
        with st.spinner("🔍 Đang tìm kiếm..."):
            try:
                rag      = load_rag_chain()
                response = rag.query(question)
                st.session_state.chat_history.append({"question": question, "response": response})
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")


def _render_rag_response(response: dict):
    """Hiển thị response RAG với badge cache."""
    # Badge cache hit
    if response.get("from_cache"):
        st.markdown('<span class="cache-badge">⚡ Từ cache — 0 token</span>', unsafe_allow_html=True)
        st.markdown("")

    answer = response.get("answer", "Không có câu trả lời").strip()
    st.markdown("**💬 Câu trả lời:**")
    st.markdown(answer)
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        source = re.sub(r"\*+", "", response.get("source", "Không xác định")).strip().replace("Nguồn:", "").strip()
        st.markdown("📄 **Nguồn:**")
        st.caption(source)
    with col2:
        confidence = response.get("confidence", "low")
        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
        conf_label = {"high": "Cao", "medium": "Trung bình", "low": "Thấp"}.get(confidence, confidence)
        st.markdown(f"{conf_emoji} **Độ tin cậy:** {conf_label}")
    with col3:
        st.markdown("🤖 **Provider:**")
        st.caption(response.get("provider", "Không xác định"))

    related = response.get("related_topics", [])
    if related:
        st.markdown("**🔗 Chủ đề liên quan:** " + " • ".join(f"`{t}`" for t in related))

    with st.expander("📋 Structured Output JSON (dùng cho UI / lưu trữ)"):
        st.json({k: v for k, v in response.items() if k != "retrieved_docs"})

    retrieved = response.get("retrieved_docs", [])
    if retrieved:
        with st.expander(f"🔍 Context đã tìm thấy ({len(retrieved)} chunks)"):
            for i, doc in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {i}** — Trang {doc.get('page','?')}, `{doc.get('filename','?')}`")
                st.text(doc.get("content", ""))
                st.divider()


# =============================================================
# TAB 2: PHÂN TÍCH ĐA PHƯƠNG THỨC (Mức 2)
# =============================================================
def render_tab_multimodal():
    st.markdown('<p class="page-title">📸 Phân tích ảnh tài liệu SGU hoặc upload video/audio (Mức 2)</p>', unsafe_allow_html=True)

    with st.expander("ℹ️ Kỹ thuật Mức 2 được áp dụng"):
        st.markdown("""
        **Visual Reasoning (Suy luận hình ảnh):**
        AI không chỉ nhận diện "đây là bảng điểm" mà còn **suy luận**: GPA có đủ học bổng không?
        Môn nào cần cải thiện? Lịch thi có trùng không?

        **Gemini File API** (cho Video/Audio):
        Thay vì encode base64 tốn băng thông, file được **upload lên Google server** một lần.
        Google lưu 48 giờ → Gemini đọc trực tiếp từ server → tiết kiệm token và băng thông.

        **Structured Extraction:**
        Tất cả kết quả trả về dưới dạng **JSON chuẩn** để code xử lý tiếp.

        **Tối ưu quota:**
        Cache theo hash ảnh → cùng ảnh + cùng câu hỏi = 0 token tốn thêm.
        """)

    input_type = st.radio(
        "Chọn loại file muốn phân tích:",
        ["🖼️ Ảnh (Visual Reasoning)", "🎥 Video/Audio (File API)"],
        horizontal=True,
    )

    if input_type == "🖼️ Ảnh (Visual Reasoning)":
        _render_image_analysis()
    else:
        _render_media_analysis()


def _render_image_analysis():
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "📁 Chọn ảnh tài liệu SGU",
            type=["jpg", "jpeg", "png", "webp"],
            help="Hỗ trợ JPG, PNG, WEBP. Tối đa 10MB.",
        )
        image_question = st.text_area(
            "❓ Câu hỏi / Yêu cầu phân tích:",
            value="Phân tích tài liệu này và đưa ra nhận xét, gợi ý hữu ích.",
            height=100,
        )

        st.markdown("**💡 Gợi ý câu hỏi Visual Reasoning:**")
        for s in [
            "Tôi có đủ điều kiện học bổng không? Môn nào cần cải thiện?",
            "Hóa đơn này hợp lệ không? Tôi đã đóng đủ học phí chưa?",
            "Lịch thi này tôi cần chuẩn bị gì? Môn nào thi sớm nhất?",
            "Thông báo này yêu cầu tôi làm gì và deadline là khi nào?",
        ]:
            if st.button(f"💬 {s[:50]}...", key=f"img_s_{s[:15]}"):
                image_question = s

        analyze_btn = st.button(
            "🔍 Phân tích ảnh (Visual Reasoning)",
            type="primary", disabled=not uploaded_file, use_container_width=True,
        )

    with col2:
        if uploaded_file:
            image_bytes = uploaded_file.read()
            st.image(image_bytes, caption=f"📎 {uploaded_file.name}", use_container_width=True)
            from multimodal import get_image_dimensions
            w, h = get_image_dimensions(image_bytes)
            st.caption(f"Kích thước: {w}×{h}px | {len(image_bytes)/1024:.1f}KB")

    if analyze_btn and uploaded_file:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        with st.spinner("🔍 Gemini đang phân tích và suy luận từ ảnh..."):
            try:
                from multimodal import analyze_image
                result = analyze_image(image_bytes, image_question)
                _render_image_result(result)
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")


def _render_image_result(result: dict):
    st.markdown("---")
    st.markdown("### 📋 Kết quả Phân tích")

    if result.get("from_cache"):
        st.markdown('<span class="cache-badge">⚡ Từ cache — 0 token</span>', unsafe_allow_html=True)
        st.markdown("")

    st.info(f"📄 **Loại tài liệu nhận diện:** {result.get('image_type', 'không xác định')}")
    st.markdown("**💬 Câu trả lời:**")
    st.markdown(result.get("answer", ""))

    reasoning = result.get("reasoning", "")
    if reasoning:
        st.markdown(
            f'<div class="reasoning-box">🧠 <b>Suy luận (Visual Reasoning):</b><br>{reasoning}</div>',
            unsafe_allow_html=True,
        )

    extracted = result.get("extracted_data", {})
    if extracted:
        with st.expander("📊 Dữ liệu Trích xuất (Structured Extraction)"):
            st.json(extracted)

    recs = result.get("recommendations", [])
    if recs:
        st.markdown("**💡 Gợi ý:**")
        for r in recs:
            st.markdown(f"- {r}")

    col1, col2 = st.columns(2)
    with col1:
        confidence = result.get("confidence", "low")
        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
        st.markdown(f"{conf_emoji} **Độ tin cậy:** {confidence.upper()}")
    with col2:
        from quota_guard import GEMINI_TEXT_MODEL
        st.markdown(f'<span class="provider-badge">🤖 {GEMINI_TEXT_MODEL} Vision</span>', unsafe_allow_html=True)

    with st.expander("📋 Xem JSON Response (Structured Output)"):
        st.json(result)


def _render_media_analysis():
    st.info("🎥 **Gemini File API**: File được upload lên Google server (không dùng base64) → Tiết kiệm băng thông và token.")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_media = st.file_uploader(
            "📁 Chọn file Video hoặc Audio",
            type=["mp4", "mov", "avi", "webm", "mp3", "wav", "m4a", "ogg"],
            help="Hỗ trợ MP4, MOV, MP3, WAV... Tối đa 100MB.",
        )
        media_question = st.text_area(
            "❓ Câu hỏi / Yêu cầu:",
            value="Tóm tắt nội dung chính và liệt kê các điểm quan trọng.",
            height=100,
        )

        for s in [
            "Tóm tắt nội dung buổi học và liệt kê các điểm cần ghi nhớ.",
            "Ai nói gì trong cuộc họp này? Liệt kê việc cần làm cho từng người.",
            "Nội dung quan trọng nhất trong video này là gì?",
        ]:
            if st.button(f"💬 {s[:50]}...", key=f"med_s_{s[:15]}"):
                media_question = s

        analyze_btn = st.button(
            "🚀 Phân tích (Gemini File API)",
            type="primary", disabled=not uploaded_media, use_container_width=True,
        )

    with col2:
        if uploaded_media:
            st.success(f"✅ Đã chọn: **{uploaded_media.name}**")
            size_mb = uploaded_media.size / 1024 / 1024
            st.metric("Kích thước file", f"{size_mb:.1f} MB")
            st.markdown("""
            **Quy trình File API:**
            1. 📤 Upload file lên Google server
            2. ⏳ Chờ Google xử lý (transcribe...)
            3. 🤖 Gemini đọc từ server → phân tích
            4. 📋 Trả về kết quả JSON
            5. 🗑️ Xóa file khỏi server
            """)
            if size_mb > 50:
                st.warning(f"⚠️ File lớn ({size_mb:.1f}MB) có thể mất 1-3 phút.")

    if analyze_btn and uploaded_media:
        ext_to_mime = {
            "mp4": "video/mp4",  "mov": "video/quicktime",
            "avi": "video/x-msvideo", "webm": "video/webm",
            "mp3": "audio/mpeg", "wav": "audio/wav",
            "m4a": "audio/mp4",  "ogg": "audio/ogg",
        }
        file_ext  = uploaded_media.name.split(".")[-1].lower()
        mime_type = ext_to_mime.get(file_ext, "video/mp4")
        file_bytes = uploaded_media.read()

        progress_bar = st.progress(0, text="📤 Đang upload lên Google File API...")
        with st.spinner("🔄 Đang xử lý... (có thể mất 1-2 phút với file lớn)"):
            try:
                from multimodal import analyze_media_file
                progress_bar.progress(30, text="⏳ Đang chờ Google xử lý file...")
                result = analyze_media_file(file_bytes, uploaded_media.name, media_question, mime_type)
                progress_bar.progress(100, text="✅ Hoàn thành!")
                _render_media_result(result)
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Lỗi: {str(e)}")


def _render_media_result(result: dict):
    st.markdown("---")
    st.markdown("### 📋 Kết quả Phân tích")

    if result.get("from_cache"):
        st.markdown('<span class="cache-badge">⚡ Từ cache — 0 token</span>', unsafe_allow_html=True)
        st.markdown("")
    st.info(f"🎬 **Loại nội dung:** {result.get('content_type', 'không xác định')}")

    if result.get("answer"):
        st.markdown("**💬 Trả lời:**")
        st.markdown(result["answer"])

    summary = result.get("summary", [])
    if summary:
        st.markdown("**📝 Tóm tắt nội dung chính:**")
        for i, point in enumerate(summary, 1):
            st.markdown(f"{i}. {point}")

    action_items = result.get("action_items", [])
    if action_items:
        st.markdown("**✅ Danh sách việc cần làm:**")
        for item in action_items:
            if isinstance(item, dict):
                label = f"- **{item.get('task', '')}**"
                if item.get("assignee"): label += f" — {item['assignee']}"
                if item.get("deadline"): label += f" — Deadline: {item['deadline']}"
                st.markdown(label)
            else:
                st.markdown(f"- {item}")

    key_moments = result.get("key_moments", [])
    if key_moments:
        st.markdown("**⏱️ Thời điểm quan trọng:**")
        for moment in key_moments:
            st.markdown(f"- {moment}")

    confidence = result.get("confidence", "low")
    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"{conf_emoji} **Độ tin cậy:** {confidence.upper()}")
    with col2:
        from quota_guard import GEMINI_TEXT_MODEL
        st.markdown(f'<span class="provider-badge">🤖 {GEMINI_TEXT_MODEL} File API</span>', unsafe_allow_html=True)

    with st.expander("📋 Xem JSON Response (Structured Output)"):
        st.json(result)


# =============================================================
# MAIN
# =============================================================
def main():
    render_sidebar()

    tab1, tab2 = st.tabs([
        "💬 Chat Tài liệu",
        "📸 Đa phương thức",
    ])
    with tab1:
        render_tab_chat()
    with tab2:
        render_tab_multimodal()


if __name__ == "__main__":
    main()
