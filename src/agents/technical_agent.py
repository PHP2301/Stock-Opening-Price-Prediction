class TechnicalAgent:
    """
    Technical Analysis Agent: Analyzes the AI models' predictions (Transformer + XGBoost)
    and standard technical indicators (RSI, MACD, Bollinger Bands) for a stock ticker.
    """
    def __init__(self):
        pass

    def analyze(self, ticker: str, trans_preds, xgb_preds, rsi_14, macd_ratio, bb_position) -> str:
        t1, t2, t3 = trans_preds
        x1, x2, x3 = xgb_preds if xgb_preds is not None else (None, None, None)
        
        report = f"--- Technical Report for {ticker} ---\n"
        report += f"1. AI Model Forecasts:\n"
        report += f"   - Transformer (Model 1) predicted return: T+1: {t1:+.4f}, T+2: {t2:+.4f}, T+3: {t3:+.4f}\n"
        if xgb_preds is not None:
            report += f"   - Hybrid XGBoost (Model 2) predicted return: T+1: {x1:+.4f}, T+2: {x2:+.4f}, T+3: {x3:+.4f}\n"
            c1, c2, c3 = x1, x2, x3
        else:
            c1, c2, c3 = t1, t2, t3

        report += f"2. Technical Indicators:\n"
        report += f"   - RSI (14): {rsi_14:.2f} (Values > 70 mean Overbought, < 30 mean Oversold)\n"
        report += f"   - MACD/Signal Ratio: {macd_ratio:.4f} (Bullish momentum if > 1.0, Bearish if < 1.0)\n"
        report += f"   - Bollinger Bands Position: {bb_position:.2f} (Upper band = 1.0, Lower band = 0.0)\n"
        
        # Formulate a bias recommendation
        threshold = 0.0050 if "VNM" in ticker.upper() else 0.0020
        if c1 > threshold and c3 > c1:
            bias = "Bullish (Strong Momentum)"
        elif c1 < -threshold:
            bias = "Bearish (Selling Pressure)"
        elif rsi_14 < 30.0:
            bias = "Bullish (Oversold Mean Reversion)"
        elif rsi_14 > 70.0:
            bias = "Bearish (Overbought mean reversion)"
        else:
            bias = "Neutral / Sideways"
            
        report += f"3. Technical Bias: {bias}\n"
        return report
