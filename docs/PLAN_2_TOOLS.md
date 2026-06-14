# BẢN KẾ HOẠCH 2: TÍCH HỢP PREDICTION MARKET TOOLS (PHASE 2 & 3)

## Mục tiêu

Áp dụng các concept chiến lược, đo lường và kiến trúc Multi-Agent từ hệ sinh thái Prediction Market (CloddsBot, PydanticAI, TradingAgents, PMXT) để biến pipeline dự báo thành một "Trading Desk" toàn diện.

## Các bước thực hiện chi tiết

### GIAI ĐOẠN 2.1: [x] Nâng cấp Backtesting (Lấy concept từ prediction-market-backtesting)
- **Tác động:** `scripts/run_backtest.py`
- [x] Bổ sung **Sharpe Ratio** để đo hiệu quả sinh lời trên rủi ro.
- [x] Bổ sung **Max Drawdown** và **Calmar Ratio**.
- [x] Bổ sung chức năng vẽ biểu đồ **Equity Curve** mô phỏng P&L (Profit & Loss).
- [x] Thêm **Strategy Signals** (từ CloddsBot):
  - [x] *Mean Reversion*: Cảnh báo khi predicted return lệch quá 1.5 độ lệch chuẩn (rolling 20-day std).
  - [x] *Momentum*: Theo dõi đà giá 3 ngày liên tiếp.

### GIAI ĐOẠN 2.2: [x] Tích hợp Multi-Agent bằng PydanticAI & TradingAgents
- **Tác động:** Tạo thư mục `src/agents/` và update `scripts/predict.py`.
- [x] **Thiết kế Architecture:**
  - [x] `technical_agent.py`: Wrap kết quả từ XGBoost + Transformer + chỉ báo kỹ thuật.
  - [x] `sentiment_agent.py`: Wrap kết quả từ news sentiment, tích hợp **PMXT** để gọi Polymarket odds (cho US stocks).
  - [x] `macro_agent.py`: Đánh giá các biến số VIX, Bond Yield, DXY.
  - [x] `risk_agent.py`: Tính toán size lệnh dựa trên ATR và MFI.
  - [x] `orchestrator.py`: Sử dụng **PydanticAI** để tạo luồng tranh luận (Bull vs Bear Debate) và ra quyết định cuối cùng (Buy/Sell/Hold) với Structured Output.

### GIAI ĐOẠN 2.3: [x] Hoàn thiện luồng Inference
- [x] Sửa lại `predict.py` để sau khi gọi `Transformer` và `XGBoost`, dữ liệu sẽ được feed thẳng vào **PydanticAI Agent**.
- [x] Agent sẽ trả về một báo cáo JSON chuẩn hóa gồm: Hành động, Độ tự tin (Confidence Score), Mức cắt lỗ (Stop Loss) và Lập luận (Reasoning).
- [x] Ghi log kết quả chi tiết ra console.

## Thời gian dự kiến: 3-4 ngày làm việc

- _Giai đoạn 2.1:_ 1 ngày
- _Giai đoạn 2.2:_ 2 ngày
- _Giai đoạn 2.3:_ 1 ngày
