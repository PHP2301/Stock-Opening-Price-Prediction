import unittest
import os
from src.agents.technical_agent import TechnicalAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.macro_agent import MacroAgent
from src.agents.risk_agent import RiskAgent
from src.agents.orchestrator import Orchestrator, TradingDecision

class TestAgentsSystem(unittest.TestCase):
    def setUp(self):
        self.tech_agent = TechnicalAgent()
        self.sent_agent = SentimentAgent()
        self.macro_agent = MacroAgent()
        self.risk_agent = RiskAgent()
        self.orchestrator = Orchestrator()

    def test_technical_agent(self):
        report = self.tech_agent.analyze(
            ticker="VNM.VN",
            trans_preds=[0.005, 0.006, 0.007],
            xgb_preds=[0.004, 0.005, 0.006],
            rsi_14=45.0,
            macd_ratio=1.1,
            bb_position=0.5
        )
        self.assertIn("Technical Report for VNM.VN", report)
        self.assertIn("Technical Bias", report)

    def test_sentiment_agent(self):
        report = self.sent_agent.analyze(
            ticker="GOOGL",
            news_sentiment_score=0.25
        )
        self.assertIn("Sentiment Report for GOOGL", report)
        self.assertIn("Sentiment Bias: Positive / Risk-On", report)

    def test_macro_agent(self):
        report = self.macro_agent.analyze(
            ticker="META",
            vix=14.5,
            bond_yield=4.2,
            usdvnd_change=0.01,
            index_return=0.15
        )
        self.assertIn("Macro Report for META", report)
        self.assertIn("Macro Bias: Bullish / Risk-On", report)

    def test_risk_agent(self):
        report = self.risk_agent.analyze(
            ticker="VNM.VN",
            close_price=68000.0,
            atr=1500.0,
            mfi=55.0
        )
        self.assertIn("Risk Report for VNM.VN", report)
        self.assertIn("Suggested Stop Loss", report)

    def test_orchestrator_debate(self):
        # We test orchestrator run in fallback mode by clearing env keys temporarily (or just let it run)
        tech_rep = self.tech_agent.analyze("VNM.VN", [0.005, 0.006, 0.007], [0.004, 0.005, 0.006], 45.0, 1.1, 0.5)
        sent_rep = self.sent_agent.analyze("VNM.VN", 0.25)
        macro_rep = self.macro_agent.analyze("VNM.VN", 14.5, 4.2, 0.01, 0.15)
        risk_rep = self.risk_agent.analyze("VNM.VN", 68000.0, 1500.0, 55.0)

        decision = self.orchestrator.run_debate(
            ticker="VNM.VN",
            close_price=68000.0,
            tech_rep=tech_rep,
            sent_rep=sent_rep,
            macro_rep=macro_rep,
            risk_rep=risk_rep
        )

        self.assertIsInstance(decision, TradingDecision)
        self.assertIn(decision.action, ["BUY", "SELL", "HOLD"])
        self.assertTrue(0.0 <= decision.confidence_score <= 1.0)
        self.assertGreaterEqual(decision.stop_loss, 0.0)
        self.assertGreaterEqual(decision.take_profit, 0.0)

if __name__ == "__main__":
    unittest.main()
