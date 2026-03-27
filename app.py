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
    .main-header {
        background: linear-gradient(135deg, #1a3a6e 0%, #2d6a9f 100%);
        padding: 20px; border-radius: 10px; color: white;
        text-align: center; margin-bottom: 20px;
    }
    .provider-badge {
        background-color: #28a745; color: white;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.85em; font-weight: bold;
    }
    .cache-badge {
        background-color: #17a2b8; color: white;
        padding: 4px 10px; border-radius: 20px;
        font-size: 0.8em; font-weight: bold;
    }
    .reasoning-box {
        background-color: #fff3cd; border-left: 4px solid #ffc107;
        padding: 12px; border-radius: 5px; margin: 8px 0;
    }
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

        st.success("✅ Gemini API Key: Có") if (
            gemini_key and gemini_key != "your_gemini_api_key_here"
        ) else st.warning("⚠️ Gemini API Key: Chưa cài")

        st.success("✅ Groq API Key: Có") if (
            groq_key and groq_key != "your_groq_api_key_here"
        ) else st.warning("⚠️ Groq API Key: Chưa cài")

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
            from quota_guard import get_counter, get_llm_limiter
            counter = get_counter()
            counts  = counter.get_all()

            # Gemini LLM
            used_llm  = counts.get("gemini_llm", 0)
            pct_llm   = min(100, int(used_llm / 1500 * 100))
            bar_color = "🟢" if pct_llm < 50 else ("🟡" if pct_llm < 80 else "🔴")
            st.markdown(f"{bar_color} **Gemini LLM**: {used_llm} / 1500 req/ngày")
            st.progress(pct_llm / 100)

            # Groq
            used_groq = counts.get("groq", 0)
            pct_groq  = min(100, int(used_groq / 14400 * 100))
            st.markdown(f"🟢 **Groq**: {used_groq} / 14400 req/ngày")
            st.progress(pct_groq / 100)

            # Vision
            used_vis  = counts.get("gemini_vision", 0)
            if used_vis > 0:
                st.markdown(f"🟢 **Gemini Vision**: {used_vis} req")

            # Cache hits
            hits = counts.get("cache_hits", 0)
            if hits > 0:
                st.success(f"⚡ Cache tiết kiệm: **{hits} request**")

            # RPM hiện tại
            limiter   = get_llm_limiter()
            rpm_now   = limiter.requests_this_minute
            rpm_color = "🟢" if rpm_now < 7 else ("🟡" if rpm_now < 9 else "🔴")
            st.caption(f"{rpm_color} Gemini RPM: {rpm_now}/10 phút này")

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
    st.markdown("### 💬 Chat với Tài liệu SGU")
    st.markdown("*Hỏi về quy định học vụ, học phí, thủ tục, học bổng...*")

    if not check_db_ready():
        st.error("⚠️ Chưa có dữ liệu! Hãy chạy: `python src/ingest.py`")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            _render_rag_response(chat["response"])

    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Câu hỏi:", placeholder="Thủ tục bảo lưu kết quả học tập như thế nào?",
            key="chat_input", label_visibility="collapsed",
        )
    with col2:
        search_btn = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)

    st.markdown("**💡 Câu hỏi mẫu:**")
    sample_cols = st.columns(3)
    samples = [
        "Học phí CNTT năm 2025?", "Điều kiện học bổng khuyến khích?",
        "Cách đóng học phí online?", "Thủ tục xin hoãn thi?",
        "Điều kiện tốt nghiệp?",   "Số điện thoại phòng Đào tạo?",
    ]
    for i, sample in enumerate(samples):
        if sample_cols[i % 3].button(sample, key=f"sample_{i}", use_container_width=True):
            question   = sample
            search_btn = True

    if (search_btn or question) and question.strip():
        with st.spinner("🔍 Đang tìm kiếm..."):
            try:
                rag      = load_rag_chain()
                response = rag.query(question)
                st.session_state.chat_history.append({"question": question, "response": response})
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    _render_rag_response(response)
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")

    if st.session_state.chat_history:
        if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


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

    with st.expander("📋 Xem JSON Response (Structured Output)"):
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
    st.markdown("### 📸 Phân tích Đa Phương Thức (Multimodal - Mức 2)")
    st.markdown("*Phân tích ảnh tài liệu SGU hoặc upload video/audio*")

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
        st.markdown('<span class="provider-badge">🤖 Gemini 2.0 Flash Vision</span>', unsafe_allow_html=True)

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
        st.markdown('<span class="provider-badge">🤖 Gemini File API</span>', unsafe_allow_html=True)

    with st.expander("📋 Xem JSON Response (Structured Output)"):
        st.json(result)


# =============================================================
# MAIN
# =============================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎓 SGU Chatbot - Hỗ trợ Sinh viên</h1>
        <p>Trường Đại học Sài Gòn</p>
    </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    tab1, tab2 = st.tabs([
        "💬 Chat Tài liệu (Mức 1)",
        "📸 Phân tích Đa phương thức (Mức 2)",
    ])
    with tab1:
        render_tab_chat()
    with tab2:
        render_tab_multimodal()


if __name__ == "__main__":
    main()