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
            reasoning = f"Giao dịch live cho {ticker_upper} đã bị tắt qua kill switch do hiệu suất lịch sử không đạt yêu cầu."
            if ticker_upper == 'VNM.VN':
                reasoning = "Giao dịch live cho VNM.VN đã bị tắt qua kill switch do hiệu suất kém (Sharpe -1.25 rolling backtest)."
            elif ticker_upper == 'META':
                reasoning = "Giao dịch live cho META tạm thời bị tắt qua kill switch (Sharpe 0.38 và MDD -38%) để chờ so sánh dữ liệu B&H."
            
            return TradingDecision(
                action="HOLD",
                confidence_score=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                debate_summary="Giao dịch cho mã này hiện đang bị tạm tắt (Kill Switch).",
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
                        "and a Bearish Analyst. You then synthesize their debate and output a structured decision."
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
                    "relative to the current Close Price."
                )

                # Synchronous run (blocking call for simplicity in pipeline)
                result = agent.run_sync(prompt)
                return result.data
            except Exception as e:
                # If LLM execution fails, drop down to the heuristic fallback
                pass

        # 2. Heuristic Fallback Mode
        # Parse biases from sub-reports
        is_tech_bull = "Bullish" in tech_rep
        is_tech_bear = "Bearish" in tech_rep
        is_sent_bull = "Positive" in sent_rep or "Risk-On" in sent_rep
        is_sent_bear = "Negative" in sent_rep or "Risk-Off" in sent_rep
        is_macro_bull = "Bullish" in macro_rep or "Risk-On" in macro_rep
        is_macro_bear = "Bearish" in macro_rep or "Risk-Off" in macro_rep

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
            reasoning = "Consensus bullish signals across Technical, Sentiment, and Macro reports."
        elif bear_votes > bull_votes and bear_votes >= 2:
            action = "SELL"
            confidence = 0.6 + 0.1 * (bear_votes - bull_votes)
            reasoning = "Consensus bearish signals across Technical, Sentiment, and Macro reports."
            stop_loss = 0.0
            take_profit = 0.0
        else:
            action = "HOLD"
            confidence = 0.5
            reasoning = "Mixed or neutral indicators across specialized reports; recommending cash preservation."
            stop_loss = 0.0
            take_profit = 0.0

        # Create debate summary
        debate = (
            f"Bullish Analyst: Argues that Technical buy triggers ({is_tech_bull}) or macro risk-on regimes ({is_macro_bull}) "
            f"justify opening a position.\n"
            f"Bearish Analyst: Counters that news headwind ({is_sent_bear}) or structural macro risks ({is_macro_bear}) "
            f"could trigger downside pressure."
        )

        return TradingDecision(
            action=action,
            confidence_score=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            debate_summary=debate,
            reasoning=reasoning
        )
