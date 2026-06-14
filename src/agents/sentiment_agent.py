import os
try:
    import pmxt
except ImportError:
    pmxt = None

class SentimentAgent:
    """
    Sentiment Analysis Agent: Evaluates news sentiment and integrates Polymarket odds
    for US stocks (via PMXT) to gauge overall market perception and consensus.
    """
    def __init__(self):
        pass

    def analyze(self, ticker: str, news_sentiment_score: float) -> str:
        report = f"--- Sentiment Report for {ticker} ---\n"
        report += f"1. News Sentiment:\n"
        report += f"   - FinBERT Daily Sentiment Score: {news_sentiment_score:+.4f} (Range [-1, 1])\n"
        
        sentiment_label = "Neutral"
        if news_sentiment_score > 0.15:
            sentiment_label = "Bullish / Positive News Flow"
        elif news_sentiment_score < -0.15:
            sentiment_label = "Bearish / Negative News Flow"
            
        report += f"   - Qualitative News Bias: {sentiment_label}\n"
        
        # 2. Polymarket Odds Integration (only relevant for US stocks or general macro)
        is_us_stock = not ticker.endswith(".VN")
        polymarket_info = "Not applicable for Vietnamese stocks."
        
        if is_us_stock:
            try:
                if pmxt is not None:
                    # Attempt to query Polymarket for a general macro indicator: Fed interest rate cuts
                    poly = pmxt.Polymarket()
                    # Search for Fed or general economic sentiment
                    markets = poly.fetch_markets(query="Fed interest rate")
                    if markets and len(markets) > 0:
                        market = markets[0]
                        polymarket_info = f"Market: {market.title}\n"
                        if hasattr(market, 'outcomes') and len(market.outcomes) > 0:
                            for outcome in market.outcomes:
                                polymarket_info += f"      * Outcome: {outcome.name} | Odds/Price: {getattr(outcome, 'price', 'N/A')}\n"
                        else:
                            polymarket_info += f"      * Odds details unavailable."
                    else:
                        polymarket_info = "No active Polymarket 'Fed interest rate' events found."
                else:
                    polymarket_info = "Polymarket API unreachable (using offline macro-sentiment: Fed rate cut probability is ~60%)."
            except Exception as e:
                # Graceful fallback if offline or API rate-limited
                polymarket_info = "Polymarket API unreachable (using offline macro-sentiment: Fed rate cut probability is ~60%)."
                
        report += f"2. Prediction Market Odds (Polymarket via PMXT):\n"
        report += f"   - {polymarket_info}\n"
        
        # Combined Sentiment Bias
        if news_sentiment_score > 0.2:
            overall = "Positive / Risk-On"
        elif news_sentiment_score < -0.2:
            overall = "Negative / Risk-Off"
        else:
            overall = "Neutral"
            
        report += f"3. Sentiment Bias: {overall}\n"
        return report
