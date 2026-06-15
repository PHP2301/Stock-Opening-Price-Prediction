class RiskAgent:
    """
    Risk Analysis Agent: Evaluates price volatility via Average True Range (ATR)
    and Money Flow Index (MFI) to formulate dynamic stop losses, take profit bounds,
    and recommended capital allocation size.
    """
    def __init__(self):
        pass

    def analyze(self, ticker: str, close_price: float, atr: float, mfi: float) -> str:
        report = f"--- Risk Report for {ticker} ---\n"
        report += f"1. Volatility & Money Flow:\n"
        report += f"   - Close Price: {close_price:,.2f}\n"
        report += f"   - ATR (Average True Range): {atr:,.2f} ({atr/close_price:.2%} of price)\n"
        report += f"   - Money Flow Index (MFI): {mfi:.2f} (Overbought > 80, Oversold < 20)\n"

        # Calculate SL / TP
        stop_loss = close_price - 2.0 * atr
        take_profit = close_price + 4.0 * atr
        
        report += f"2. Proposed Order Guardrails (Long Position):\n"
        report += f"   - Suggested Stop Loss (2.0 ATR): {stop_loss:,.2f}\n"
        report += f"   - Suggested Take Profit (4.0 ATR): {take_profit:,.2f}\n"
        report += f"   - Risk-Reward Ratio: 1:2.0\n"

        # Capital Allocation recommendation
        if mfi > 80.0:
            size_pct = 2.0  # Reduce size (overbought)
            comment = "Reduce exposure due to extremely high Money Flow (MFI > 80)"
        elif mfi < 20.0:
            size_pct = 7.5  # Increase size (oversold)
            comment = "Increase exposure due to oversold Money Flow (MFI < 20)"
        else:
            size_pct = 5.0  # Standard risk size
            comment = "Standard risk allocation"
            
        report += f"3. Risk Recommendation:\n"
        report += f"   - Position Size Recommendation: {size_pct:.1f}% of total capital\n"
        report += f"   - Commentary: {comment}\n"
        return report

    def calculate_position_size(self, confidence_score: float, win_rate_history: float,
                                close_price: float, stop_loss: float, take_profit: float,
                                current_drawdown: float = 0.0) -> dict:
        """
        Calculates optimal position size using the Kelly Criterion.
        Uses p = min(confidence_score, win_rate_history) to prevent overleveraging.
        """
        denom = close_price - stop_loss
        num = take_profit - close_price
        R = num / denom if denom > 0 else 2.00
        
        p = min(confidence_score, win_rate_history)
        
        kelly_raw = 0.0
        kelly_half = 0.0
        kelly_size_pct = 0.0
        
        if R > 0:
            # Kelly Formula: f* = (p * R - (1 - p)) / R
            kelly_raw = (p * R - (1.0 - p)) / R
            kelly_fraction = max(kelly_raw, 0.0) / 2.0
            
            # Scale down Kelly size based on current portfolio drawdown
            if current_drawdown < -0.25:
                kelly_fraction *= 0.25
            elif current_drawdown < -0.15:
                kelly_fraction *= 0.5
                
            kelly_half = kelly_fraction
            kelly_size_pct = min(kelly_half, 0.25) * 100.0
            
        return {
            "p": p,
            "R": R,
            "kelly_raw": kelly_raw,
            "kelly_half": kelly_half,
            "kelly_size_pct": kelly_size_pct
        }
