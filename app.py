# =============================================================
# app.py - STREAMLIT UI CHÍNH (3 TAB)
# =============================================================
# Tab 1: Chat tài liệu - RAG Pipeline (Level 1)
# Tab 2: Phân tích ảnh - Multimodal (Level 2)
# Tab 3: Agent thông minh - ReAct Agent (Level 3)
#
# Chạy ứng dụng: streamlit run app.py
#
# Lưu ý Token Management:
# - st.cache_resource: Cache RAGChain, Agent → không khởi tạo lại mỗi interaction
# - Hiển thị provider đang dùng (Gemini/Groq/Ollama) để báo cáo Token Management
# - Hiển thị số token ước tính (dựa trên độ dài response)
# =============================================================

import streamlit as st
import json
import os
import sys
from pathlib import Path

# Thêm thư mục src vào path để import các module
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# =============================================================
# CẤU HÌNH TRANG STREAMLIT
# =============================================================
st.set_page_config(
    page_title="SGU Chatbot - Hỗ trợ Sinh viên",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tùy chỉnh giao diện
st.markdown("""
<style>
    /* Header chính */
    .main-header {
        background: linear-gradient(135deg, #1a3a6e 0%, #2d6a9f 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Badge provider đang dùng */
    .provider-badge {
        background-color: #28a745;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
    }
    /* Box câu trả lời */
    .answer-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1a3a6e;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    /* Badge confidence */
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    /* Source box */
    .source-box {
        background-color: #e8f4f8;
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 0.9em;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# KHỞI TẠO COMPONENT (cache để tái dùng - Token Management)
# =============================================================
# st.cache_resource: Cache object qua các lần re-run của Streamlit
# Tránh khởi tạo lại RAGChain/Agent mỗi lần người dùng click

@st.cache_resource(show_spinner="🔄 Đang khởi tạo RAG Chain...")
def load_rag_chain():
    """Load RAGChain một lần, cache cho toàn session."""
    from rag_chain import RAGChain
    return RAGChain()

@st.cache_resource(show_spinner="🤖 Đang khởi tạo Agent...")
def load_agent():
    """Load Agent một lần, cache cho toàn session."""
    from agent import SGUAgent
    return SGUAgent()


def check_db_ready() -> bool:
    """Kiểm tra ChromaDB đã được tạo chưa."""
    from ingest import check_db_exists
    return check_db_exists()


# =============================================================
# SIDEBAR - Thông tin hệ thống và cài đặt
# =============================================================
def render_sidebar():
    """Render sidebar với thông tin và cài đặt."""
    with st.sidebar:
        st.markdown("## 🎓 SGU Chatbot")
        st.markdown("*Hỗ trợ Sinh viên ĐH Sài Gòn*")
        st.divider()
        
        # Trạng thái hệ thống
        st.markdown("### 📊 Trạng thái hệ thống")
        
        # Kiểm tra DB
        db_ready = check_db_ready()
        if db_ready:
            st.success("✅ ChromaDB: Sẵn sàng")
        else:
            st.error("❌ ChromaDB: Chưa có dữ liệu")
            st.info("Chạy: `python src/ingest.py`")
        
        # Kiểm tra API keys
        from dotenv import load_dotenv
        load_dotenv()
        
        gemini_key = os.getenv("GOOGLE_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            st.success("✅ Gemini API Key: Có")
        else:
            st.warning("⚠️ Gemini API Key: Chưa cài")
        
        if groq_key and groq_key != "your_groq_api_key_here":
            st.success("✅ Groq API Key: Có")
        else:
            st.warning("⚠️ Groq API Key: Chưa cài")
        
        st.divider()
        
        # Token Management info
        st.markdown("### 🔢 Token Management")
        from llm_router import get_current_provider, get_llm_info
        current = get_current_provider()
        
        if current != "Chưa khởi tạo":
            st.markdown(f'<span class="provider-badge">🤖 {current}</span>', 
                       unsafe_allow_html=True)
        
        # Hiển thị quota thông tin
        with st.expander("📋 Quota các Provider"):
            info = get_llm_info()
            for provider, details in info.items():
                st.markdown(f"**{provider}**")
                st.markdown(f"- Quota: {details['quota']}")
                st.markdown(f"- Tốc độ: {details['speed']}")
                st.markdown(f"- Chi phí: {details['cost']}")
                st.divider()
        
        st.divider()
        
        # Hướng dẫn nhanh
        st.markdown("### 📖 Hướng dẫn")
        st.markdown("""
        **Bước 1:** Đặt file PDF vào thư mục `data/`  
        **Bước 2:** Chạy `python src/ingest.py`  
        **Bước 3:** Chạy `streamlit run app.py`  
        
        **Tài liệu:** Cẩm nang SV SGU 2022
        """)
        
        st.divider()
        st.markdown("*v1.0 | LangChain + Gemini + ChromaDB*")


# =============================================================
# TAB 1: CHAT TÀI LIỆU (RAG - Level 1)
# =============================================================
def render_tab_chat():
    """
    Tab 1: Giao diện chat với RAG Pipeline.
    Người dùng nhập câu hỏi, hệ thống tìm context và trả lời.
    """
    st.markdown("### 💬 Chat với Tài liệu SGU")
    st.markdown("*Hỏi về quy định học vụ, học phí, thủ tục, học bổng...*")
    
    # Kiểm tra DB
    if not check_db_ready():
        st.error("⚠️ Chưa có dữ liệu! Hãy chạy: `python src/ingest.py`")
        st.info("Đặt file PDF vào thư mục `data/` rồi chạy lệnh trên.")
        return
    
    # Khởi tạo lịch sử chat trong session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # --- Hiển thị lịch sử chat ---
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            _render_rag_response(chat["response"])
    
    # --- Input câu hỏi ---
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Câu hỏi của bạn:",
            placeholder="Ví dụ: Thủ tục bảo lưu kết quả học tập như thế nào?",
            key="chat_input",
            label_visibility="collapsed",
        )
    
    with col2:
        search_btn = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)
    
    # Câu hỏi mẫu (giúp người dùng test nhanh)
    st.markdown("**💡 Câu hỏi mẫu:**")
    sample_cols = st.columns(3)
    samples = [
        "Học phí CNTT năm 2025?",
        "Điều kiện học bổng khuyến khích?",
        "Cách đóng học phí online?",
        "Thủ tục xin hoãn thi?",
        "Điều kiện tốt nghiệp?",
        "Số điện thoại phòng Đào tạo?",
    ]
    for i, sample in enumerate(samples):
        col_idx = i % 3
        if sample_cols[col_idx].button(sample, key=f"sample_{i}", use_container_width=True):
            question = sample
            search_btn = True
    
    # --- Xử lý câu hỏi ---
    if (search_btn or question) and question.strip():
        with st.spinner("🔍 Đang tìm kiếm..."):
            try:
                rag = load_rag_chain()
                response = rag.query(question)
                
                # Lưu vào lịch sử
                st.session_state.chat_history.append({
                    "question": question,
                    "response": response,
                })
                
                # Hiển thị kết quả mới nhất
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    _render_rag_response(response)
                    
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    # Nút xóa lịch sử
    if st.session_state.chat_history:
        if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


def _render_rag_response(response: dict):
    """
    Hiển thị response JSON dưới dạng UI đẹp.
    
    Lý do dùng st.markdown thay vì HTML div cho answer:
    - LLM trả về text có thể chứa markdown (*bold*, xuống dòng...)
    - Nhúng vào HTML div sẽ bị hiển thị sai (ô trắng hoặc ký tự lạ)
    - st.markdown tự xử lý markdown đúng cách
    """
    import re

    # Lấy và làm sạch câu trả lời
    answer = response.get("answer", "Không có câu trả lời")
    # Xóa markdown ** nếu LLM trả về text thô lẫn vào answer
    answer_clean = answer.strip()

    # Hiển thị câu trả lời bằng st.markdown (hỗ trợ xuống dòng, bold...)
    st.markdown("**💬 Câu trả lời:**")
    st.markdown(answer_clean)
    st.divider()

    # Metadata trong columns
    col1, col2, col3 = st.columns(3)

    with col1:
        # Lấy source và làm sạch markdown thừa (**Nguồn:** ...)
        source = response.get("source", "Không xác định")
        source = re.sub(r"\*+", "", source).strip()  # Xóa ** nếu có
        source = source.replace("Nguồn:", "").strip()  # Xóa prefix thừa
        st.markdown("📄 **Nguồn:**")
        st.caption(source)

    with col2:
        # Độ tin cậy với màu sắc
        confidence = response.get("confidence", "low")
        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
        conf_label = {"high": "Cao", "medium": "Trung bình", "low": "Thấp"}.get(confidence, confidence)
        st.markdown(f"{conf_emoji} **Độ tin cậy:** {conf_label}")

    with col3:
        # Provider đang dùng (yêu cầu báo cáo Token Management)
        provider = response.get("provider", "Không xác định")
        st.markdown(f"🤖 **Provider:**")
        st.caption(provider)

    # Chủ đề liên quan
    related = response.get("related_topics", [])
    if related:
        st.markdown("**🔗 Chủ đề liên quan:** " + " • ".join(f"`{t}`" for t in related))

    # Expand xem raw JSON (cho báo cáo Structured Output)
    with st.expander("📋 Xem JSON Response (Structured Output)"):
        display_response = {k: v for k, v in response.items() if k != "retrieved_docs"}
        st.json(display_response)

    # Expand xem các chunk context đã tìm được
    retrieved = response.get("retrieved_docs", [])
    if retrieved:
        with st.expander(f"🔍 Context đã tìm thấy ({len(retrieved)} chunks)"):
            for i, doc in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {i}** — Trang {doc.get('page','?')}, `{doc.get('filename','?')}`")
                st.text(doc.get("content", ""))
                st.divider()


# =============================================================
# TAB 2: PHÂN TÍCH ẢNH (Multimodal - Level 2)
# =============================================================
def render_tab_image():
    """
    Tab 2: Upload ảnh tài liệu và phân tích với Gemini Vision.
    Use cases: Hóa đơn học phí, bảng điểm, lịch thi...
    """
    st.markdown("### 📸 Phân tích Ảnh Tài liệu")
    st.markdown("*Upload ảnh hóa đơn học phí, bảng điểm, lịch thi để phân tích*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Upload ảnh
        uploaded_file = st.file_uploader(
            "📁 Chọn ảnh tài liệu",
            type=["jpg", "jpeg", "png", "webp"],
            help="Hỗ trợ JPG, PNG, WEBP. Kích thước tối đa 10MB."
        )
        
        # Câu hỏi về ảnh
        image_question = st.text_area(
            "❓ Câu hỏi về ảnh:",
            value="Đây là loại tài liệu gì? Hãy trích xuất thông tin quan trọng.",
            height=100,
            help="Mô tả bạn muốn biết gì về ảnh này"
        )
        
        # Câu hỏi mẫu cho ảnh
        st.markdown("**💡 Gợi ý câu hỏi:**")
        image_samples = [
            "Tôi đã đóng học phí chưa và số tiền là bao nhiêu?",
            "Liệt kê tất cả môn học và điểm trong bảng điểm này",
            "Môn thi nào có lịch thi sớm nhất?",
        ]
        for sample in image_samples:
            if st.button(f"💬 {sample}", key=f"img_sample_{sample[:20]}"):
                image_question = sample
        
        analyze_btn = st.button("🔍 Phân tích ảnh", type="primary", 
                                disabled=not uploaded_file,
                                use_container_width=True)
    
    with col2:
        if uploaded_file:
            # Hiển thị ảnh đã upload
            image_bytes = uploaded_file.read()
            st.image(image_bytes, caption=f"📎 {uploaded_file.name}", use_container_width=True)
            
            # Thông tin ảnh
            from multimodal import get_image_dimensions
            w, h = get_image_dimensions(image_bytes)
            st.caption(f"Kích thước: {w}×{h}px | Size: {len(image_bytes)/1024:.1f}KB")
    
    # --- Phân tích ảnh ---
    if analyze_btn and uploaded_file:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        
        with st.spinner("🔍 Gemini Vision đang phân tích ảnh..."):
            try:
                from multimodal import analyze_image
                result = analyze_image(image_bytes, image_question)
                
                st.markdown("---")
                st.markdown("### 📋 Kết quả Phân tích")
                
                # Loại tài liệu
                img_type = result.get("image_type", "không xác định")
                st.info(f"📄 **Loại tài liệu nhận diện:** {img_type}")
                
                # Câu trả lời
                answer = result.get("answer", "")
                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True
                )
                
                # Dữ liệu trích xuất
                extracted = result.get("extracted_data", {})
                if extracted:
                    with st.expander("📊 Dữ liệu Trích xuất"):
                        st.json(extracted)
                
                # Lưu ý quan trọng
                notes = result.get("important_notes", [])
                if notes:
                    st.markdown("**⚠️ Lưu ý:**")
                    for note in notes:
                        st.markdown(f"- {note}")
                
                # Độ tin cậy và Provider
                col_conf, col_prov = st.columns(2)
                with col_conf:
                    confidence = result.get("confidence", "low")
                    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
                    st.markdown(f"{conf_emoji} Độ tin cậy: **{confidence.upper()}**")
                with col_prov:
                    # Multimodal luôn dùng Gemini Vision
                    st.markdown('<span class="provider-badge">🤖 Gemini 2.0 Flash Vision</span>',
                               unsafe_allow_html=True)
                
                # Raw JSON
                with st.expander("📋 Xem JSON Response (Structured Output)"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"❌ Lỗi phân tích ảnh: {str(e)}")
                if "API_KEY" in str(e):
                    st.info("💡 Cần cài GOOGLE_API_KEY trong file .env")


# =============================================================
# TAB 3: AGENT THÔNG MINH (Level 3)
# =============================================================
def render_tab_agent():
    """
    Tab 3: Agent ReAct có thể kết hợp nhiều tool.
    Phù hợp với câu hỏi phức tạp cần tính toán + tìm kiếm.
    """
    st.markdown("### 🤖 Agent Thông minh SGU")
    st.markdown("*Agent có thể tìm kiếm tài liệu, tính GPA, kiểm tra học bổng...*")
    
    # Giải thích các tool
    with st.expander("🛠️ Các công cụ của Agent"):
        tool_data = {
            "Tool": ["search_document", "calculate_gpa", "get_current_date", "check_scholarship"],
            "Mô tả": [
                "Tìm kiếm trong tài liệu SGU (ChromaDB)",
                "Tính GPA hệ 4 từ điểm hệ 10",
                "Lấy ngày tháng hiện tại",
                "Kiểm tra điều kiện học bổng"
            ],
            "Ví dụ Input": [
                '"điều kiện tốt nghiệp"',
                '"Toán:8:3,Lý:7:2,Anh:9:3"',
                '"" (không cần input)',
                '"gpa:3.2,credits:18,failures:0,discipline:không"'
            ],
        }
        st.table(tool_data)
    
    # Kiểm tra DB
    if not check_db_ready():
        st.warning("⚠️ search_document tool cần ChromaDB. Chạy `python src/ingest.py` trước.")
    
    # Input câu hỏi phức tạp
    agent_question = st.text_area(
        "🤔 Câu hỏi phức tạp (Agent có thể kết hợp nhiều tool):",
        placeholder="Ví dụ: Tính GPA cho tôi: Toán:8.5:3, Lý:7:2, Anh:9:3. Tôi có đủ điều kiện học bổng không? Và deadline nộp hồ sơ học bổng là khi nào?",
        height=120,
    )
    
    # Câu hỏi mẫu cho Agent
    st.markdown("**💡 Câu hỏi mẫu cho Agent:**")
    agent_samples = [
        "Tính GPA: Toán:8.5:3, Lý:7.0:2, Anh:9.0:3, CSDL:8.0:3. Có đủ học bổng không?",
        "Hôm nay là ngày mấy? Tìm thông tin về lịch đóng học phí.",
        "Điều kiện xét tốt nghiệp tại SGU là gì?",
        "Tôi học 18TC, GPA 3.5, không có môn nào dưới 5. Tôi được học bổng mấy loại?",
    ]
    
    for sample in agent_samples:
        if st.button(f"💬 {sample[:60]}...", key=f"agent_{sample[:20]}"):
            agent_question = sample
    
    run_btn = st.button(
        "🚀 Chạy Agent", 
        type="primary", 
        disabled=not agent_question.strip(),
        use_container_width=True
    )
    
    # --- Chạy Agent ---
    if run_btn and agent_question.strip():
        with st.spinner("🤖 Agent đang xử lý... (có thể mất 10-30 giây)"):
            try:
                agent = load_agent()
                result = agent.run(agent_question)
                
                st.markdown("---")
                st.markdown("### 📋 Kết quả Agent")
                
                # Câu trả lời
                answer = result.get("answer", "Không có câu trả lời")
                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True
                )
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tools_used = result.get("tools_used", [])
                    if tools_used:
                        st.markdown(f"🛠️ **Tools đã dùng:**")
                        for t in tools_used:
                            st.markdown(f"  - `{t}`")
                
                with col2:
                    source = result.get("source", "Agent (nhiều nguồn)")
                    st.markdown(
                        f'<div class="source-box">📄 **Nguồn:** {source}</div>',
                        unsafe_allow_html=True
                    )
                
                with col3:
                    confidence = result.get("confidence", "medium")
                    provider = result.get("provider", "Không xác định")
                    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
                    st.markdown(f"{conf_emoji} **Confidence:** {confidence.upper()}")
                    st.markdown(
                        f'<span class="provider-badge">🤖 {provider}</span>',
                        unsafe_allow_html=True
                    )
                
                # Raw JSON
                with st.expander("📋 Xem JSON Response (Structured Output)"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"❌ Agent gặp lỗi: {str(e)}")
                st.info("💡 Thử câu hỏi đơn giản hơn hoặc dùng Tab 1 (Chat tài liệu)")


# =============================================================
# HÀM MAIN - Render toàn bộ UI
# =============================================================
def main():
    """Hàm main render toàn bộ ứng dụng Streamlit."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 SGU Chatbot - Hỗ trợ Sinh viên</h1>
        <p>Trường Đại học Sài Gòn | RAG + LangChain + Gemini</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # 3 Tab chính
    tab1, tab2, tab3 = st.tabs([
        "💬 Chat Tài liệu (RAG)",
        "📸 Phân tích Ảnh",
        "🤖 Agent Thông minh"
    ])
    
    with tab1:
        render_tab_chat()
    
    with tab2:
        render_tab_image()
    
    with tab3:
        render_tab_agent()
    
    # Footer
    st.divider()
    st.markdown(
        "<center><small>📚 Dữ liệu: Sổ tay Hỗ trợ Sinh viên SGU 2022 + Học phí 2025-2026 | "
        "🔧 Stack: LangChain + Gemini 2.0 Flash + ChromaDB + Streamlit</small></center>",
        unsafe_allow_html=True
    )


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    main()