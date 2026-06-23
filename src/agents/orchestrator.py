import os
import re
from pydantic import BaseModel, Field
try:
    from pydantic_ai import Agent
except ImportError:
    Agent = None

class TradingDecision(BaseModel):
    action: str = Field(description="Final trading decision: BUY, SELL, or HOLD")
    confidence_score: float = Field(description="Confidence score between 0.0 (low) and 1.0 (high)")
    stop_loss: float = Field(description="Recommended Stop Loss price level (0 if holding)")
    take_profit: float = Field(description="Recommended Take Profit price level (0 if holding)")
    debate_summary: str = Field(description="Key arguments from the simulated Bullish and Bearish analysts debate")
    reasoning: str = Field(description="Final combined rationale for the action taken")

class Orchestrator:
    """
    Master Orchestrator Agent: Takes reports from Technical, Sentiment, Macro,
    and Risk agents, simulates a Bull vs Bear debate using PydanticAI (or a
    heuristic fallback if API keys are missing), and outputs a structured TradingDecision.
    """
    def __init__(self):
        pass

    def run_debate(self, ticker: str, close_price: float, tech_rep: str, sent_rep: str, macro_rep: str, risk_rep: str, forecast_return: float = 0.0) -> TradingDecision:
        # Tỷ lệ giao dịch cho từng mã
        TRADING_ENABLED = {
            'VNM.VN': True,    # Sharpe cải thiện rõ rệt, MDD -2.52% sau nâng cấp
            'GOOGL':  True,    # Sharpe 1.39 — triển khai
            'META':   True,    # Triển khai kèm bộ lọc bảo vệ vốn Kelly & Trailing Stop
        }

        # Cho phép ghi đè/bỏ qua kill switch bằng biến môi trường (ví dụ: $env:OVERRIDE_KILL_SWITCH="1")
        override_kill_switch = os.environ.get("OVERRIDE_KILL_SWITCH", "0") == "1"

        ticker_upper = ticker.upper()
        if not override_kill_switch and not TRADING_ENABLED.get(ticker_upper, True):
            reasoning = f"Tính năng tự động giao dịch cho {ticker_upper} tạm thời bị khóa vì hiệu suất đầu tư lịch sử không ổn định."
            if ticker_upper == 'VNM.VN':
                reasoning = "Giao dịch tự động cho VNM.VN tạm thời bị khóa nhằm bảo vệ tài sản do hiệu suất sinh lời quá thấp so với mức rủi ro trong quá khứ."
            elif ticker_upper == 'META':
                reasoning = "Giao dịch tự động cho META tạm thời bị khóa để tránh rủi ro sụt giảm tài sản lớn, hệ thống đang tạm ngưng quan sát thêm."
            
            return TradingDecision(
                action="HOLD",
                confidence_score=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                debate_summary="Chế độ tự động giao dịch cho mã này hiện đang bị khóa để bảo vệ tài khoản (Kill Switch).",
                reasoning=reasoning
            )

        # Trích xuất bias từ các báo cáo phụ trước (dùng cho cả LLM và Fallback)
        def extract_bias_line(report_text, label):
            for line in report_text.splitlines():
                if label in line:
                    return line
            return ""

        tech_bias_line = extract_bias_line(tech_rep, "Technical Bias:")
        sent_bias_line = extract_bias_line(sent_rep, "Sentiment Bias:")
        macro_bias_line = extract_bias_line(macro_rep, "Macro Bias:")
        risk_line = extract_bias_line(risk_rep, "Risk Level:")

        is_tech_bull = "Bullish" in tech_bias_line
        is_tech_bear = "Bearish" in tech_bias_line
        is_sent_bull = "Positive" in sent_bias_line or "Risk-On" in sent_bias_line
        is_sent_bear = "Negative" in sent_bias_line or "Risk-Off" in sent_bias_line
        is_macro_bull = "Bullish" in macro_bias_line or "Risk-On" in macro_bias_line
        is_macro_bear = "Bearish" in macro_bias_line or "Risk-Off" in macro_bias_line

        bull_votes = sum([is_tech_bull, is_sent_bull, is_macro_bull])
        bear_votes = sum([is_tech_bear, is_sent_bear, is_macro_bear])
        risk_upper = risk_line.upper()

        # 1. Check if LLM API Key is configured
        model_name = None
        if os.environ.get("GEMINI_API_KEY"):
            model_name = "gemini-1.5-flash"
        elif os.environ.get("OPENAI_API_KEY"):
            model_name = "openai:gpt-4o-mini"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            model_name = "anthropic:claude-3-5-sonnet-latest"

        if model_name is not None and Agent is not None:
            try:
                # Initialize PydanticAI Agent
                agent = Agent(
                    model=model_name,
                    result_type=TradingDecision,
                    system_prompt=(
                        "You are an expert investment orchestrator. You analyze four specialized stock reports "
                        "(Technical, Sentiment, Macro, Risk) and conduct a simulated debate between a Bullish Analyst "
                        "and a Bearish Analyst. You then synthesize their debate and output a structured decision. "
                        "IMPORTANT: You must write the 'debate_summary' and 'reasoning' in clear, simple Vietnamese "
                        "suitable for non-technical users (avoid using complex financial or technical jargon directly, "
                        "explain metrics like Sharpe or Kill Switch in simple terms if you refer to them)."
                    )
                )

                prompt = (
                    f"Ticker: {ticker}\n"
                    f"Current Close Price: {close_price:,.2f}\n\n"
                    f"{tech_rep}\n"
                    f"{sent_rep}\n"
                    f"{macro_rep}\n"
                    f"{risk_rep}\n\n"
                    "Simulate a debate between 'Bullish Analyst' and 'Bearish Analyst', synthesize, and output "
                    "the final TradingDecision structure. Ensure Stop Loss and Take Profit are mathematically sound "
                    "relative to the current Close Price. Both 'debate_summary' and 'reasoning' MUST be written in Vietnamese."
                )

                # Synchronous run (blocking call for simplicity in pipeline)
                result = agent.run_sync(prompt)
                decision = result.data
                # Áp dụng kịch bản quyết đoán (Tất tay / Bán chắc chắn bao lời) cho kết quả LLM
                if bull_votes >= 3 and "HIGH" not in risk_upper and "CRITICAL" not in risk_upper and "EXTREME" not in risk_upper:
                    decision.action = "BUY"
                    decision.confidence_score = 0.99
                    decision.reasoning = "🔥 CƠ HỘI VÀNG (ALL-IN): Đồng thuận tuyệt đối 3/3 chuyên gia cùng mức rủi ro an toàn. Tín hiệu cực kỳ chắc chắn để gom hàng tất tay!"
                elif bear_votes >= 3:
                    decision.action = "SELL"
                    decision.confidence_score = 0.99
                    decision.reasoning = "🚨 KHẨN CẤP (BÁN CHẮC CHẮN BAO LỜI): Sự đồng thuận giảm giá tuyệt đối từ 3/3 chuyên gia. Xu hướng đảo chiều đã rõ như ban ngày, khuyến nghị bán tháo/chốt lời sạch vị thế ngay lập tức!"
                    decision.stop_loss = 0.0
                    decision.take_profit = 0.0
                else:
                    # Điều chỉnh độ tự tin dựa trên độ lớn và hướng của dự báo lợi nhuận tương lai (forecast_return)
                    boost = 0.0
                    if decision.action == "BUY" and forecast_return > 0:
                        boost = min(forecast_return * 2, 0.15)
                    elif decision.action == "SELL" and forecast_return < 0:
                        boost = min(abs(forecast_return) * 2, 0.15)
                    
                    decision.confidence_score += boost
                    decision.confidence_score = min(decision.confidence_score, 1.0)
                
                # Thêm thông báo dự báo tương lai
                if forecast_return > 0:
                    decision.reasoning += (
                        f" Mô hình AI dự báo giá có thể tăng khoảng "
                        f"{forecast_return*100:.1f}% trong giai đoạn tới."
                    )
                elif forecast_return < 0:
                    decision.reasoning += (
                        f" Mô hình AI dự báo giá có thể giảm khoảng "
                        f"{abs(forecast_return)*100:.1f}% trong giai đoạn tới."
                    )
                
                return decision
            except Exception as e:
                # If LLM execution fails, drop down to the heuristic fallback
                pass

        # 2. Heuristic Fallback Mode
        # Điều chỉnh trọng số MUA/BÁN dựa trên mức độ rủi ro (Risk Level)
        if "CRITICAL" in risk_upper or "EXTREME" in risk_upper:
            bull_votes = 0
            bear_votes += 2
        elif "HIGH" in risk_upper:
            bull_votes = max(0, bull_votes - 1)
            bear_votes += 1
        elif "LOW" in risk_upper or "MINIMAL" in risk_upper:
            bull_votes += 1
            bear_votes = max(0, bear_votes - 1)

        # Parse proposed stop_loss / take_profit from risk_rep
        sl_match = re.search(r"Suggested Stop Loss.*:\s*([0-9,.]+)", risk_rep)
        tp_match = re.search(r"Suggested Take Profit.*:\s*([0-9,.]+)", risk_rep)
        
        try:
            stop_loss = float(sl_match.group(1).replace(",", "")) if sl_match else close_price * 0.95
            take_profit = float(tp_match.group(1).replace(",", "")) if tp_match else close_price * 1.10
        except Exception:
            stop_loss = close_price * 0.95
            take_profit = close_price * 1.10

        # 3. Phân loại quyết định giao dịch theo phiếu bầu
        if bull_votes >= 3 and "HIGH" not in risk_upper and "CRITICAL" not in risk_upper and "EXTREME" not in risk_upper:
            # Kịch bản MUA cực kỳ chắc chắn (Tất tay / All-in)
            action = "BUY"
            confidence = 0.99
            reasoning = "🔥 CƠ HỘI VÀNG (ALL-IN): Đồng thuận tuyệt đối 3/3 chuyên gia cùng mức rủi ro an toàn. Tín hiệu cực kỳ chắc chắn để gom hàng tất tay!"
        elif bear_votes >= 3:
            # Kịch bản BÁN cực kỳ chắc chắn (Chốt lời bao lời / Bán tháo khẩn cấp)
            action = "SELL"
            confidence = 0.99
            reasoning = "🚨 KHẨN CẤP (BÁN CHẮC CHẮN BAO LỜI): Sự đồng thuận giảm giá tuyệt đối từ 3/3 chuyên gia. Xu hướng đảo chiều đã rõ như ban ngày, khuyến nghị bán tháo/chốt lời sạch vị thế ngay lập tức!"
            stop_loss = 0.0
            take_profit = 0.0
        elif bull_votes > bear_votes and bull_votes >= 2:
            action = "BUY"
            confidence = min(0.95, 0.5 + 0.15 * (bull_votes - bear_votes))
            reasoning = f"Tín hiệu MUA đồng thuận từ các chuyên gia (Bull: {bull_votes}, Bear: {bear_votes})."
            if "HIGH" in risk_upper:
                confidence = max(0.4, confidence - 0.20)
                reasoning += " Lưu ý: Mức rủi ro CAO, khuyến nghị đi kèm giảm tỷ trọng."
        elif bear_votes > bull_votes and bear_votes >= 2:
            action = "SELL"
            confidence = min(0.95, 0.5 + 0.15 * (bear_votes - bull_votes))
            reasoning = f"Tín hiệu BÁN đồng thuận từ các chuyên gia (Bear: {bear_votes}, Bull: {bull_votes})."
            if "CRITICAL" in risk_upper or "EXTREME" in risk_upper:
                confidence = min(0.99, confidence + 0.15)
                reasoning += " Cảnh báo: Rủi ro CỰC KỲ NGHIÊM TRỌNG, ưu tiên thoát vị thế bảo vệ tài sản."
            stop_loss = 0.0
            take_profit = 0.0
        else:
            # Xử lý các tình huống giằng co hoặc thiếu đồng thuận
            action = "HOLD"
            confidence = 0.5
            if bull_votes == bear_votes and bull_votes > 0:
                reasoning = f"Thị trường giằng co cực mạnh (Bull: {bull_votes} vs Bear: {bear_votes}). Khuyến nghị đứng ngoài quan sát."
            elif "HIGH" in risk_upper or "CRITICAL" in risk_upper:
                reasoning = f"Hệ thống khóa giao dịch (HOLD) do mức rủi ro quá cao ({risk_line.strip() if ':' in risk_line else risk_line})."
            else:
                reasoning = "Không đủ số lượng phiếu đồng thuận (tối thiểu 2 phiếu cùng chiều) từ các chuyên gia để mở vị thế."
            stop_loss = 0.0
            take_profit = 0.0

        # Điều chỉnh độ tự tin dựa trên độ lớn và hướng của dự báo lợi nhuận tương lai (forecast_return)
        boost = 0.0
        if action == "BUY" and forecast_return > 0:
            boost = min(forecast_return * 2, 0.15)
        elif action == "SELL" and forecast_return < 0:
            boost = min(abs(forecast_return) * 2, 0.15)

        confidence += boost
        confidence = min(confidence, 1.0)

        # Thêm thông báo dự báo tương lai
        if forecast_return > 0:
            reasoning += (
                f" Mô hình AI dự báo giá có thể tăng khoảng "
                f"{forecast_return*100:.1f}% trong giai đoạn tới."
            )
        elif forecast_return < 0:
            reasoning += (
                f" Mô hình AI dự báo giá có thể giảm khoảng "
                f"{abs(forecast_return)*100:.1f}% trong giai đoạn tới."
            )

        # Tạo tóm tắt tranh luận tiếng Việt đơn giản dễ hiểu
        debate = (
            f"Phân tích Tăng giá (Bullish): Cho rằng tín hiệu kỹ thuật tốt ({'Đúng' if is_tech_bull else 'Sai'}) hoặc vĩ mô ổn định ({'Đúng' if is_macro_bull else 'Sai'}) ủng hộ việc mua vào.\n"
            f"Phân tích Giảm giá (Bearish): Cảnh báo rằng tin tức xấu ({'Có' if is_sent_bear else 'Không'}) hoặc rủi ro vĩ mô lớn ({'Có' if is_macro_bear else 'Không'}) có thể gây áp lực giảm giá."
        )

        return TradingDecision(
            action=action,
            confidence_score=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            debate_summary=debate,
            reasoning=reasoning
        )
