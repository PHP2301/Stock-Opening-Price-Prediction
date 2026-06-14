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

    def run_debate(self, ticker: str, close_price: float, tech_rep: str, sent_rep: str, macro_rep: str, risk_rep: str) -> TradingDecision:
        # Tỷ lệ giao dịch cho từng mã
        TRADING_ENABLED = {
            'VNM.VN': False,   # Sharpe -1.25 rolling — không triển khai
            'GOOGL':  True,    # Sharpe 1.39 — triển khai
            'META':   False,   # Sharpe 0.38 + MDD -38% — cần thêm dữ liệu B&H trước khi quyết
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
                return result.data
            except Exception as e:
                # If LLM execution fails, drop down to the heuristic fallback
                pass

        # 2. Heuristic Fallback Mode
        # Parse biases from sub-reports (only the specific bias lines to avoid matching text in parentheses)
        def extract_bias_line(report_text, label):
            for line in report_text.splitlines():
                if label in line:
                    return line
            return ""

        tech_bias_line = extract_bias_line(tech_rep, "Technical Bias:")
        sent_bias_line = extract_bias_line(sent_rep, "Sentiment Bias:")
        macro_bias_line = extract_bias_line(macro_rep, "Macro Bias:")

        is_tech_bull = "Bullish" in tech_bias_line
        is_tech_bear = "Bearish" in tech_bias_line
        is_sent_bull = "Positive" in sent_bias_line or "Risk-On" in sent_bias_line
        is_sent_bear = "Negative" in sent_bias_line or "Risk-Off" in sent_bias_line
        is_macro_bull = "Bullish" in macro_bias_line or "Risk-On" in macro_bias_line
        is_macro_bear = "Bearish" in macro_bias_line or "Risk-Off" in macro_bias_line

        bull_votes = sum([is_tech_bull, is_sent_bull, is_macro_bull])
        bear_votes = sum([is_tech_bear, is_sent_bear, is_macro_bear])

        # Parse proposed stop_loss / take_profit from risk_rep
        sl_match = re.search(r"Suggested Stop Loss.*:\s*([0-9,.]+)", risk_rep)
        tp_match = re.search(r"Suggested Take Profit.*:\s*([0-9,.]+)", risk_rep)
        
        try:
            stop_loss = float(sl_match.group(1).replace(",", "")) if sl_match else close_price * 0.95
            take_profit = float(tp_match.group(1).replace(",", "")) if tp_match else close_price * 1.10
        except Exception:
            stop_loss = close_price * 0.95
            take_profit = close_price * 1.10

        if bull_votes > bear_votes and bull_votes >= 2:
            action = "BUY"
            confidence = 0.6 + 0.1 * (bull_votes - bear_votes)
            reasoning = "Đồng thuận tín hiệu MUA vào từ tất cả các chuyên gia (Kỹ thuật tốt, Tin tức tích cực và Vĩ mô ổn định)."
        elif bear_votes > bull_votes and bear_votes >= 2:
            action = "SELL"
            confidence = 0.6 + 0.1 * (bear_votes - bull_votes)
            reasoning = "Đồng thuận tín hiệu BÁN ra từ các chuyên gia (Kỹ thuật xấu, Tin tức tiêu cực và Vĩ mô bất ổn)."
            stop_loss = 0.0
            take_profit = 0.0
        else:
            action = "HOLD"
            confidence = 0.5
            reasoning = "Tín hiệu thị trường đang trái chiều và chưa rõ ràng; khuyến nghị tiếp tục đứng ngoài quan sát để bảo toàn dòng tiền."
            stop_loss = 0.0
            take_profit = 0.0

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
