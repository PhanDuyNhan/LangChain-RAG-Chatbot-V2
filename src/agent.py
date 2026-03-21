# =============================================================
# agent.py - LANGCHAIN REACT AGENT VỚI CUSTOM TOOLS (LEVEL 3)
# =============================================================
# Agent dùng chiến lược ReAct (Reasoning + Acting):
#   1. REASONING: Suy nghĩ về câu hỏi cần làm gì
#   2. ACTION: Gọi tool phù hợp
#   3. OBSERVATION: Xem kết quả tool
#   4. Lặp lại cho đến khi có câu trả lời cuối cùng
#
# Các Tool được đăng ký:
#   1. search_document(query)  → Tìm trong ChromaDB
#   2. calculate_gpa(data)     → Tính GPA hệ 4
#   3. get_current_date()      → Ngày hôm nay
#   4. check_scholarship(data) → Kiểm tra điều kiện học bổng
#
# Structured Output: Agent trả về JSON chuẩn
#   {"answer": ..., "tools_used": [...], "reasoning": ..., "source": ...}
#
# Lý do dùng Agent thay vì RAG đơn thuần:
# - RAG chỉ tìm và trả lời, không tính toán được
# - Agent có thể kết hợp nhiều tool (tìm → tính → kiểm tra)
# - Phù hợp với câu hỏi phức tạp: "Tính GPA của tôi và xem tôi có đủ điều kiện học bổng không"
# =============================================================

import os
import json
import re
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_router import get_llm
from ingest import (
    COLLECTION_NAME, CHROMA_DB_PATH,
    check_db_exists, load_vector_store, create_embeddings
)

load_dotenv()


# =============================================================
# ĐỊNH NGHĨA CÁC TOOL
# =============================================================
# Dùng decorator @tool của LangChain để đăng ký hàm thành tool
# Docstring của hàm = mô tả tool cho Agent biết khi nào nên dùng

@tool
def search_document(query: str) -> str:
    """
    Tìm kiếm thông tin trong tài liệu Cẩm nang Sinh viên SGU.
    Dùng tool này khi cần tìm quy định, thủ tục, học phí, học bổng,
    thông tin liên hệ, hoặc bất kỳ thông tin nào trong tài liệu SGU.
    
    Input: Câu hỏi hoặc từ khóa tìm kiếm (tiếng Việt)
    Output: Các đoạn văn liên quan từ tài liệu, có kèm số trang
    """
    try:
        # Tái dùng embedding và vector store đã có
        embeddings = create_embeddings()
        vector_store = load_vector_store(embeddings)
        
        # Tìm top 3 chunk liên quan (ít hơn RAG chain để tiết kiệm token)
        docs = vector_store.similarity_search(query, k=3)
        
        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        
        # Format kết quả
        results = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            filename = doc.metadata.get("filename", "Tài liệu SGU")
            results.append(f"[Nguồn {i} - Trang {page}, {filename}]\n{doc.page_content}")
        
        return "\n\n".join(results)
    
    except Exception as e:
        return f"Lỗi khi tìm kiếm: {str(e)}"


@tool
def calculate_gpa(data: str) -> str:
    """
    Tính điểm trung bình tích lũy (GPA) hệ 4 theo công thức của SGU.
    
    Input format (dạng chuỗi): "TênMôn:điểmHệ10:sốTínChỉ,TênMôn:điểmHệ10:sốTínChỉ,..."
    Ví dụ: "Toán:8.5:3,Lý:7.0:2,Hóa:6.5:3"
    
    Nếu input là JSON string, cũng chấp nhận dạng:
    '[{"name":"Toán","score":8.5,"credits":3},...]'
    
    Output: JSON với GPA hệ 4, xếp loại, chi tiết từng môn
    """
    # Bảng quy đổi điểm hệ 10 → hệ 4 theo quy định SGU/Bộ GD
    # Theo Thông tư 08/2021/TT-BGDĐT
    def score_to_gpa4(score_10: float) -> float:
        """Quy đổi điểm hệ 10 sang hệ 4."""
        if score_10 >= 9.0:  return 4.0   # A+
        if score_10 >= 8.5:  return 3.7   # A
        if score_10 >= 8.0:  return 3.5   # B+
        if score_10 >= 7.0:  return 3.0   # B
        if score_10 >= 6.5:  return 2.5   # C+
        if score_10 >= 5.5:  return 2.0   # C
        if score_10 >= 5.0:  return 1.5   # D+
        if score_10 >= 4.0:  return 1.0   # D
        return 0.0                         # F
    
    def gpa4_to_rank(gpa4: float) -> str:
        """Xếp loại tốt nghiệp theo GPA hệ 4."""
        if gpa4 >= 3.6:  return "Xuất sắc"
        if gpa4 >= 3.2:  return "Giỏi"
        if gpa4 >= 2.5:  return "Khá"
        if gpa4 >= 2.0:  return "Trung bình"
        return "Không đạt (dưới 2.0)"
    
    try:
        courses = []
        
        # Thử parse dạng JSON trước
        if data.strip().startswith("["):
            raw = json.loads(data)
            for item in raw:
                courses.append({
                    "name": item.get("name", "Môn học"),
                    "score10": float(item.get("score", 0)),
                    "credits": int(item.get("credits", 3)),
                })
        else:
            # Parse dạng "TênMôn:điểm:TC,..."
            for entry in data.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                parts = entry.split(":")
                if len(parts) != 3:
                    continue
                name = parts[0].strip()
                score = float(parts[1].strip())
                credits = int(parts[2].strip())
                courses.append({"name": name, "score10": score, "credits": credits})
        
        if not courses:
            return "Không parse được dữ liệu. Dùng format: 'TênMôn:điểm:TC,...'"
        
        # Tính GPA theo công thức SGU:
        # GPA = Σ(điểmHệ4_i × tínChỉ_i) / Σ(tínChỉ_i)
        total_weighted = 0.0
        total_credits = 0
        details = []
        
        for c in courses:
            gpa4 = score_to_gpa4(c["score10"])
            weighted = gpa4 * c["credits"]
            total_weighted += weighted
            total_credits += c["credits"]
            details.append({
                "môn": c["name"],
                "điểm_10": c["score10"],
                "điểm_4": gpa4,
                "tín_chỉ": c["credits"],
            })
        
        if total_credits == 0:
            return "Tổng tín chỉ = 0, không thể tính GPA."
        
        final_gpa = round(total_weighted / total_credits, 2)
        rank = gpa4_to_rank(final_gpa)
        
        result = {
            "gpa_he_4": final_gpa,
            "tong_tin_chi": total_credits,
            "xep_loai": rank,
            "chi_tiet": details,
            "ghi_chu": "Không tính GDTC và GDQP vào GPA theo quy định SGU"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"Lỗi tính GPA: {str(e)}. Format đúng: 'TênMôn:điểm:TC,...'"


@tool
def get_current_date() -> str:
    """
    Lấy ngày tháng năm hiện tại theo giờ Việt Nam.
    Dùng khi cần xác định: deadline đóng học phí, hạn nộp đơn,
    thời gian còn lại đến kỳ thi, hoặc bất kỳ câu hỏi về thời gian.
    
    Output: Chuỗi ngày tháng dạng "Thứ X, DD/MM/YYYY HH:MM (giờ Việt Nam)"
    """
    now = datetime.now()
    
    # Tên thứ tiếng Việt
    weekdays = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm",
                "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    weekday = weekdays[now.weekday()]
    
    return (
        f"{weekday}, {now.strftime('%d/%m/%Y %H:%M')} (giờ máy chủ)\n"
        f"Năm học hiện tại: 2024-2025\n"
        f"Học kỳ thường: HK1 (tháng 9-1), HK2 (tháng 2-6), HK phụ (tháng 6-8)"
    )


@tool
def check_scholarship(data: str) -> str:
    """
    Kiểm tra điều kiện xét học bổng khuyến khích học tập tại SGU.
    
    Input format: "gpa:X.X,credits:N,failures:N,discipline:có/không"
    Ví dụ: "gpa:3.2,credits:18,failures:0,discipline:không"
    
    Tham số:
    - gpa: Điểm trung bình hệ 4 của học kỳ xét
    - credits: Số tín chỉ đã đăng ký trong HK đó
    - failures: Số môn dưới 5.0 (hệ 10)
    - discipline: Có bị kỷ luật không (có/không)
    
    Output: JSON kết quả xét học bổng theo từng mức
    """
    try:
        # Parse input
        params = {}
        for item in data.split(","):
            item = item.strip()
            if ":" in item:
                key, val = item.split(":", 1)
                params[key.strip()] = val.strip()
        
        gpa = float(params.get("gpa", 0))
        credits = int(params.get("credits", 0))
        failures = int(params.get("failures", 0))
        discipline = params.get("discipline", "không").lower()
        
        # ====================================================
        # TIÊU CHÍ HỌC BỔNG SGU (theo KB029 + KB028F)
        # ====================================================
        results = {
            "thong_tin_dau_vao": {
                "gpa_hk": gpa,
                "so_tin_chi": credits,
                "so_mon_duoi_5": failures,
                "bi_ky_luat": discipline
            },
            "dieu_kien_co_ban": {},
            "ket_qua": [],
            "khuyen_nghi": ""
        }
        
        # Điều kiện cơ bản (tất cả loại học bổng đều phải đạt)
        basic_conditions = {
            "Đăng ký ≥ 15 tín chỉ": credits >= 15,
            "Không có môn dưới 5.0": failures == 0,
            "GPA ≥ 6.0 (hệ 10) / ≥ 2.0 (hệ 4)": gpa >= 2.0,
            "Không bị kỷ luật": discipline in ["không", "no", "false", "n"],
        }
        results["dieu_kien_co_ban"] = {k: ("✅ Đạt" if v else "❌ Chưa đạt") 
                                        for k, v in basic_conditions.items()}
        
        all_basic_met = all(basic_conditions.values())
        
        if not all_basic_met:
            results["ket_qua"] = ["❌ Không đủ điều kiện cơ bản xét học bổng"]
            results["khuyen_nghi"] = (
                "Cần đáp ứng TẤT CẢ điều kiện cơ bản: "
                "≥15 tín chỉ, không môn nào dưới 5, GPA≥2.0, không kỷ luật."
            )
            return json.dumps(results, ensure_ascii=False, indent=2)
        
        # Xét từng mức học bổng (theo KB028F - Top % của ngành)
        # Lưu ý: % chính xác phụ thuộc thứ hạng trong ngành
        # Ở đây xét theo ngưỡng GPA hệ 4
        scholarship_levels = [
            {
                "ten": "Học bổng Xuất sắc (Top 3% - Miễn 100% HP + tiền mặt)",
                "dieu_kien_gpa": gpa >= 3.6,
                "mo_ta": "GPA ≥ 3.6 (hệ 4) - Xuất sắc",
            },
            {
                "ten": "Học bổng Loại 1 (Top 4% - Miễn 100% HP)",
                "dieu_kien_gpa": gpa >= 3.2,
                "mo_ta": "GPA ≥ 3.2 (hệ 4) - Giỏi",
            },
            {
                "ten": "Học bổng Loại 2 (Top 5% - Giảm 50% HP)",
                "dieu_kien_gpa": gpa >= 2.8,
                "mo_ta": "GPA ≥ 2.8 (hệ 4)",
            },
            {
                "ten": "Học bổng Khuyến khích (Top 10%)",
                "dieu_kien_gpa": gpa >= 2.5,
                "mo_ta": "GPA ≥ 2.5 (hệ 4) - Khá",
            },
        ]
        
        eligible = []
        for level in scholarship_levels:
            if level["dieu_kien_gpa"]:
                eligible.append(f"✅ Có thể xét: {level['ten']}")
        
        if not eligible:
            results["ket_qua"] = [
                f"❌ GPA {gpa} chưa đủ ngưỡng học bổng (tối thiểu ≥ 2.5 hệ 4)"
            ]
            results["khuyen_nghi"] = "Cần nâng GPA lên ít nhất 2.5 để xét học bổng khuyến khích."
        else:
            results["ket_qua"] = eligible
            results["khuyen_nghi"] = (
                "Kết quả học bổng phụ thuộc vào thứ hạng trong ngành. "
                "Liên hệ Phòng HTDN-HTSV để biết chính xác. "
                "Email: hotrosinhvien@sgu.edu.vn | ĐT: (028) 39 381 901"
            )
        
        return json.dumps(results, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return f"Lỗi kiểm tra học bổng: {str(e)}. Format: 'gpa:X.X,credits:N,failures:N,discipline:không'"


# =============================================================
# AGENT CHÍNH
# =============================================================

# System prompt cho Agent - khác với RAG chain vì Agent cần biết cách dùng tool
AGENT_SYSTEM_PROMPT = """Bạn là trợ lý thông minh hỗ trợ sinh viên Trường Đại học Sài Gòn (SGU).
Bạn có thể sử dụng các công cụ (tools) để trả lời câu hỏi.

Sau khi có đủ thông tin, hãy trả lời theo định dạng JSON sau:
{{"answer": "Câu trả lời đầy đủ", "tools_used": ["tool1", "tool2"], "source": "nguồn thông tin", "confidence": "high/medium/low"}}

Luôn trả lời bằng tiếng Việt. Nếu cần tính toán, hãy dùng tool calculate_gpa hoặc check_scholarship."""


class SGUAgent:
    """
    Agent thông minh có thể kết hợp nhiều tool để trả lời câu hỏi phức tạp.
    Dùng kiến trúc ReAct: Lý luận → Hành động → Quan sát → Lặp lại
    """
    
    def __init__(self):
        """Khởi tạo Agent với danh sách tool và LLM."""
        # Danh sách tool đăng ký với Agent
        self.tools = [
            search_document,
            calculate_gpa,
            get_current_date,
            check_scholarship,
        ]
        
        # Lấy LLM (ưu tiên Gemini, fallback Groq/Ollama)
        self.llm, self.provider_name = get_llm()
        
        # Tạo Agent dùng ReAct prompt từ LangChain Hub
        # ReAct = Reasoning + Acting: LLM lý luận trước khi gọi tool
        try:
            # Pull ReAct prompt chuẩn từ LangChain Hub
            react_prompt = hub.pull("hwchase17/react")
            self.agent = create_react_agent(self.llm, self.tools, react_prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,          # Hiển thị quá trình suy nghĩ (debug)
                max_iterations=5,      # Tối đa 5 lần gọi tool (tránh vòng lặp vô hạn)
                handle_parsing_errors=True,  # Xử lý lỗi parse gracefully
            )
        except Exception as e:
            raise RuntimeError(f"Không thể tạo Agent: {e}")
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Chạy Agent với câu hỏi đầu vào.
        
        Args:
            question: Câu hỏi phức tạp của sinh viên
        
        Returns:
            Dict JSON chuẩn với answer, tools_used, source, confidence, provider
        """
        tools_used = []
        
        try:
            # Thêm context về ngày tháng vào câu hỏi
            enriched_question = (
                f"{question}\n\n"
                f"(Lưu ý: Trả lời bằng tiếng Việt. "
                f"Dùng JSON format: {{\"answer\":...,\"tools_used\":[...],\"source\":...,\"confidence\":...}})"
            )
            
            result = self.agent_executor.invoke({"input": enriched_question})
            raw_output = result.get("output", "")
            
            # Parse JSON từ output của Agent
            parsed = self._parse_agent_output(raw_output)
            parsed["provider"] = self.provider_name
            
            return parsed
        
        except Exception as e:
            return {
                "answer": f"Agent gặp lỗi: {str(e)[:200]}\n\nHãy thử đặt câu hỏi đơn giản hơn hoặc dùng tab Chat.",
                "tools_used": tools_used,
                "source": "Lỗi Agent",
                "confidence": "low",
                "provider": self.provider_name,
            }
    
    def _parse_agent_output(self, raw: str) -> dict:
        """Parse JSON từ output của Agent, xử lý các trường hợp đặc biệt."""
        # Xóa markdown
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        
        # Tìm JSON trong text (Agent đôi khi thêm text trước/sau JSON)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "answer": data.get("answer", text),
                    "tools_used": data.get("tools_used", []),
                    "source": data.get("source", "Agent (nhiều nguồn)"),
                    "confidence": data.get("confidence", "medium"),
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback: text không có JSON
        return {
            "answer": raw,
            "tools_used": [],
            "source": "Agent",
            "confidence": "medium",
        }


# Singleton pattern cho Agent
_agent_instance = None

def get_agent() -> SGUAgent:
    """Trả về Agent instance (singleton) để tránh khởi tạo lại."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SGUAgent()
    return _agent_instance


# =============================================================
# TEST NHANH - chạy: python src/agent.py
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST AGENT SGU")
    print("=" * 60)
    
    # Test các tool trực tiếp trước
    print("\n--- Test Tool: calculate_gpa ---")
    result = calculate_gpa.invoke("Toán:8.5:3,Lý:7.0:2,Hóa:6.5:3,Anh:9.0:3")
    print(result)
    
    print("\n--- Test Tool: check_scholarship ---")
    result = check_scholarship.invoke("gpa:3.2,credits:18,failures:0,discipline:không")
    print(result)
    
    print("\n--- Test Tool: get_current_date ---")
    result = get_current_date.invoke("")
    print(result)
    
    print("\n--- Test Agent (câu hỏi phức tạp) ---")
    agent = SGUAgent()
    result = agent.run("Tính GPA cho tôi: Toán:8:3, Lý:7:2, Anh:9:3. Tôi có đủ điều kiện học bổng không?")
    print(json.dumps(result, ensure_ascii=False, indent=2))
