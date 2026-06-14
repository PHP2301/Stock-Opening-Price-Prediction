# 📘 HƯỚNG DẪN TOÀN DIỆN: FEATURES & TOOLS

> Tài liệu này giải thích chi tiết:
> 1. Hệ thống Features hiện tại (34) và 8 Features mới sắp thêm (→ 42)
> 2. 11 Prediction Market Tools được đánh giá
> 3. **Mối quan hệ bổ sung giá trị** giữa Features và Tools

---

## MỤC LỤC

- [Phần A — Hệ Thống Features](#phần-a--hệ-thống-features)
- [Phần B — 11 Prediction Market Tools](#phần-b--11-prediction-market-tools)
- [Phần C — Bổ Sung Giá Trị Lẫn Nhau](#phần-c--bổ-sung-giá-trị-lẫn-nhau)

---

# PHẦN A — HỆ THỐNG FEATURES

## A1. Features hiện tại (34 đặc trưng)

### Nhánh 1 — Giá & Động lượng (12 features)

| # | Feature | Ý nghĩa | Tại sao cần? |
|---|---|---|---|
| 1 | `gap_open` | Gap mở cửa qua đêm | Đo sentiment overnight — gap lớn = tin tốt/xấu ngoài giờ |
| 2 | `open_return` | Return giữa 2 phiên mở cửa | Đo xu hướng mở cửa riêng biệt với đóng cửa |
| 3 | `buying_pressure` | Áp lực mua cuối ngày | Close gần High = phe mua thắng; gần Low = phe bán thắng |
| 4 | `shadow_ratio` | Tỷ lệ râu nến trên/dưới | Râu trên dài = lực bán mạnh tại đỉnh |
| 5 | `intraday_range` | Biên độ dao động trong phiên | Phiên rộng = biến động mạnh, tính bất ổn |
| 6-8 | `return_1d/2d/3d` | Return 1-2-3 ngày trước | Momentum ngắn hạn — đà tăng/giảm gần đây |
| 9-11 | `mom_5d/10d/20d` | Momentum 5-10-20 ngày | Xu hướng trung hạn — phát hiện sóng |
| 12 | `dist_ma50` | Khoảng cách tới MA50 | Xa MA50 = overextended, khả năng mean-revert |

### Nhánh 2 — Khối lượng & Biến động (6 features)

| # | Feature | Ý nghĩa | Tại sao cần? |
|---|---|---|---|
| 13 | `volume_change` | Thay đổi volume so hôm trước | Volume đột biến = sự kiện bất thường |
| 14 | `volume_sma_ratio` | Volume / SMA20 | > 1.5 = volume spike, institutional activity |
| 15 | `volume_zscore` | Z-score volume 20 ngày | Chuẩn hóa mức độ bất thường của volume |
| 16 | `ad_line_ratio` | A/D Line chuẩn hóa | Phân phối vs tích lũy — smart money flow |
| 17 | `obv_zscore` | Z-score OBV delta 5 ngày | Dòng tiền tích lũy bất thường |
| 18 | `vol_ratio` | Biến động 5d / 60d | > 1 = thị trường đang nóng hơn bình thường |

### Nhánh 3 — Kỹ thuật, Vĩ mô & Lịch (16 features)

| # | Feature | Ý nghĩa | Tại sao cần? |
|---|---|---|---|
| 19 | `rsi_14` | Relative Strength Index | < 30 oversold, > 70 overbought |
| 20 | `macd_ratio` | MACD / Signal | Crossover = signal mua/bán |
| 21 | `bb_position` | Vị trí trong Bollinger Bands | 0 = Lower band, 1 = Upper band |
| 22 | `adx_14` | Average Directional Index | > 25 = trending mạnh, < 20 = sideway |
| 23 | `stoch_k` | Stochastic %K | Momentum quá mua/bán ngắn hạn |
| 24 | `efficiency_ratio` | Hiệu quả xu hướng | 1 = trending hoàn hảo, 0 = choppy |
| 25 | `vix_lag1` | VIX (Fear Index) | Cao = thị trường sợ hãi, thường giảm |
| 26 | `bond_yield_lag1` | US 10Y Bond Yield | Tăng → equity giảm (inverse correlation) |
| 27 | `usdvnd_change` | Biến động USD/VND | Ảnh hưởng dòng vốn ngoại vào VN |
| 28 | `vnindex_return_lag1` | Return VNINDEX/NASDAQ | Sentiment thị trường chung |
| 29-30 | `day_of_week_sin/cos` | Ngày trong tuần (cyclical) | Thứ 2/6 có pattern riêng (Monday effect) |
| 31-32 | `month_sin/cos` | Tháng (cyclical) | Hiệu ứng "Sell in May", January effect |
| 33 | `is_quarter_end` | Cuối quý | Rebalancing quỹ → volume spike |
| 34 | `days_before_tet` | Cận Tết Nguyên Đán | VNM-specific: rally trước Tết |

---

## A2. Features MỚI sắp thêm (8 features → tổng 42)

### Nhánh 4 — Dòng tiền & Cổ tức (8 features mới)

| # | Feature | Ý nghĩa | Nguồn dữ liệu | Tại sao cần? |
|---|---|---|---|---|
| 35 | `mfi_14` | Money Flow Index 14 ngày | `pandas_ta.mfi()` | RSI có trọng số volume — phát hiện divergence chính xác hơn RSI thuần |
| 36 | `dividend_flag` | 1 nếu ngày ex-right, 0 nếu không | yfinance (US) / vnstock events (VN) | Giá thường giảm đúng ngày ex-right bằng mức cổ tức |
| 37 | `days_to_dividend` | Số ngày đến kỳ chia cổ tức tiếp | yfinance / vnstock | Gần ngày chốt quyền → rally (dividend capture) |
| 38 | `days_after_dividend` | Số ngày kể từ kỳ chia gần nhất | yfinance / vnstock | Sau ex-date → sell-off (bán chốt lời cổ tức) |
| 39 | `foreign_net_buy_proxy` | Proxy khối ngoại mua/bán ròng | `volume_zscore × CLV` | Volume bất thường + đóng gần high = institutional buying |
| 40 | `foreign_net_buy_5d` | Rolling 5d của proxy | Tính từ #39 | Xu hướng ngắn hạn dòng tiền khối ngoại |
| 41 | `foreign_net_buy_20d` | Rolling 20d của proxy | Tính từ #39 | Xu hướng dài hạn — accumulation/distribution |
| 42 | `self_net_buy_proxy` | Proxy tự doanh | MFI divergence + large volume | Phát hiện hoạt động tự doanh CTCK |

> **Ghi chú:** Foreign Net Buy thực và Self Net Buy thực không lấy được miễn phí từ API nào (vnstock, VNDirect, TCBS, SSI đều chặn). Dùng proxy tính từ OHLCV.

---

# PHẦN B — 11 PREDICTION MARKET TOOLS

## B1. Bảng tổng quan

| # | Tool | Loại | Mục đích chính | Phù hợp? |
|---|---|---|---|---|
| 1 | **Polymarket Dataset** | Dữ liệu | 107GB, 1.1 tỷ trades lịch sử prediction market | ⚠️ Tham khảo |
| 2 | **prediction-market-backtesting** | Framework | Backtesting PnL, Sharpe, Drawdown | ✅ Lấy concept |
| 3 | **Polybot** | Analytics | Reverse-engineer chiến lược trader | ✅ Lấy concept |
| 4 | **polymarket_lp_tool** | Bot | Liquidity provision trên Polymarket | ❌ Không phù hợp |
| 5 | **PolyWeather** | Bot | Dự báo thời tiết cho prediction market | ❌ Không phù hợp |
| 6 | **CloddsBot** | Bot | 118+ strategies tự động (arbitrage, mean reversion...) | ✅ Lấy strategies |
| 7 | **PydanticAI** | Framework | Build AI agents production-ready, structured output | ✅ **Dùng trực tiếp** |
| 8 | **TradingAgents** | Framework | Multi-agent trading desk (analyst/risk/fund manager) | ✅ **Dùng trực tiếp** |
| 9 | **PMXT** | SDK | Unified API cho Polymarket, Kalshi (CCXT-style) | ⚠️ Tham khảo |
| 10 | **PM Bot Toolkits** | Toolkit | Copy trading, whale alerts, arbitrage | ⚠️ Lấy concept |
| 11 | **Awesome List** | Tài liệu | Tổng hợp 100+ tools prediction market | ✅ Reference |

## B2. Chi tiết từng tool

### 1️⃣ Polymarket Dataset (107GB)
- **Repo:** `SII-WANGZJ/Polymarket_data`
- **Nội dung:** 1.1 tỷ trades từ blockchain Polygon, format parquet
- **Cách dùng:** `pip install huggingface_hub` → `huggingface-cli download SII-WANGZJ/Polymarket_data`
- **Áp dụng:** Lọc earnings events GOOGL/META → dùng odds làm sentiment feature

### 2️⃣ prediction-market-backtesting
- **Repo:** `evan-kolberg/prediction-market-backtesting`
- **Nội dung:** Extension NautilusTrader cho backtesting prediction markets
- **Metrics:** Equity curve, P&L ticks, Sharpe ratio, Max Drawdown, Brier score
- **Áp dụng:** Copy metrics (Sharpe, Drawdown, Calmar) vào `run_backtest.py`

### 3️⃣ Polybot
- **Repo:** `ent0n29/polybot`
- **Nội dung:** Phân tích wallet trader, phát hiện pattern giao dịch
- **Áp dụng:** Concept "reverse-engineer whale" → cải thiện `foreign_net_buy_proxy` bằng volume + CLV pattern

### 4️⃣ polymarket_lp_tool ❌
- **Repo:** `lihanyu81/polymarket_lp_tool`
- **Nội dung:** Bot đặt limit orders hai bên trên Polymarket để thu spread
- **Không phù hợp:** Chỉ cho market making trên prediction market, không liên quan stock

### 5️⃣ PolyWeather ❌
- **Repo:** `yangyuan-zhen/PolyWeather`
- **Nội dung:** AI phân tích METAR/SPECI cho weather prediction markets
- **Không phù hợp:** Chỉ cho weather prediction, không liên quan stock

### 6️⃣ CloddsBot (118+ strategies)
- **Repo:** `alsk1992/CloddsBot`
- **Nội dung:** AI agent (Claude) tự trade 1000+ markets với 118+ strategies
- **Strategies áp dụng được:**
  - **Mean Reversion** → khi predicted return quá extreme → signal đảo chiều
  - **Momentum** → khi 3 ngày cùng hướng → signal tiếp tục
  - **Confidence Filter** → chỉ trade khi nhiều model đồng ý

### 7️⃣ PydanticAI ⭐
- **Repo:** `pydantic/pydantic-ai`
- **Cài đặt:** `pip install pydantic-ai`
- **Nội dung:** Framework Python xây dựng AI agents với structured output, type-safe
- **Áp dụng trực tiếp:** Orchestrator cho multi-agent predict pipeline, đảm bảo output nhất quán

### 8️⃣ TradingAgents ⭐
- **Repo:** `TauricResearch/TradingAgents`
- **Nội dung:** Mô phỏng trading desk chuyên nghiệp bằng multi-agent LLM
- **Kiến trúc:** Fundamental Analyst → Technical Analyst → Sentiment Analyst → Bull/Bear Debate → Risk Manager → Fund Manager
- **Áp dụng trực tiếp:** Wrap Transformer + XGBoost vào Technical Analyst agent

### 9️⃣ PMXT
- **Repo:** `pmxt-dev/pmxt`
- **Cài đặt:** `pip install pmxt`
- **Nội dung:** Unified API cho Polymarket, Kalshi, Limitless (giống CCXT cho crypto)
- **Áp dụng:** Lấy Polymarket odds cho earnings GOOGL/META → sentiment feature

### 🔟 PM Bot Toolkits
- **Repo:** `PredictionXBT/PredictOS`
- **Nội dung:** Copy trading, whale alerts, cross-platform arbitrage
- **Áp dụng:** Concept whale detection → cải thiện volume anomaly proxy

### 1️⃣1️⃣ Awesome List
- **Repo:** `marvinrailey-git/awesome-polymarket-tools`
- **Nội dung:** Tổng hợp 100+ tools, SDKs, dashboards, research
- **Áp dụng:** Reference khi cần tìm tool mới

---

# PHẦN C — BỔ SUNG GIÁ TRỊ LẪN NHAU

## C1. Sơ đồ quan hệ

```
╔══════════════════════════════════════════════════════════════════════╗
║                    CHUỖI GIÁ TRỊ TỔNG THỂ                         ║
║                                                                     ║
║  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐     ║
║  │  DỮ LIỆU    │───▶│  FEATURES    │───▶│  PREDICTION        │     ║
║  │  (Tools)     │    │  (42 cols)   │    │  (Models + Agents) │     ║
║  └─────────────┘    └──────────────┘    └────────────────────┘     ║
║        │                   │                      │                 ║
║        ▼                   ▼                      ▼                 ║
║  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐     ║
║  │ Polymarket   │    │ Backtesting  │    │  Multi-Agent       │     ║
║  │ PMXT, Data   │    │ Framework    │    │  PydanticAI +      │     ║
║  │ (nguồn mới)  │    │ (đánh giá)   │    │  TradingAgents     │     ║
║  └─────────────┘    └──────────────┘    └────────────────────┘     ║
╚══════════════════════════════════════════════════════════════════════╝
```

## C2. Bảng ma trận bổ sung giá trị

| | Features (42) | Tools |
|---|---|---|
| **Features → Tools** | Features cung cấp **input data** cho các tool xử lý | — |
| **Tools → Features** | — | Tools cung cấp **nguồn dữ liệu mới** và **phương pháp tính** cho features |

### Chi tiết từng mối quan hệ:

#### 🔗 Tools bổ sung cho Features:

| Tool | Bổ sung Feature nào | Cách bổ sung |
|---|---|---|
| **Polybot** (whale analysis) | → `foreign_net_buy_proxy` | Cung cấp **phương pháp** phát hiện whale: volume spike + CLV pattern |
| **PM Toolkits** (whale alerts) | → `self_net_buy_proxy` | Concept large-block detection → proxy institutional activity |
| **PMXT** (Polymarket odds) | → Feature mới `earnings_sentiment` | Cung cấp **nguồn dữ liệu** xác suất earnings beat/miss |
| **CloddsBot** (strategies) | → Strategy signals | Mean reversion + momentum → signal mua/bán dựa trên features |
| **Backtesting framework** | → Đánh giá chất lượng features | Sharpe, Drawdown → đo features nào thực sự có giá trị |

#### 🔗 Features bổ sung cho Tools:

| Feature | Bổ sung Tool nào | Cách bổ sung |
|---|---|---|
| 34 features kỹ thuật | → **TradingAgents** (Technical Analyst) | Agent dùng features làm input phân tích |
| `mfi_14` + volume features | → **PydanticAI** (Risk Agent) | Agent đánh giá rủi ro dựa trên MFI divergence |
| `dividend_flag` + timing | → **CloddsBot** strategies | Dividend capture strategy: mua trước ex-date, bán sau |
| `foreign_net_buy_proxy` | → **TradingAgents** (Sentiment Agent) | Agent detect institutional flow → bullish/bearish bias |
| VIX, Bond Yield, DXY | → **TradingAgents** (Macro Analyst) | Agent phân tích macro environment |

## C3. Giá trị kết hợp theo Phase

### Phase 1: Features → Nền tảng cho mọi thứ
```
42 Features ────────────────────────────────────────────────────
   │                                                          │
   │  MFI, Dividend, Foreign Proxy = DỮ LIỆU MỚI             │
   │  → Transformer + XGBoost HỌC TỐT HƠN                    │
   │  → Prediction accuracy TĂNG                              │
   │                                                          │
   └──────────────────────────────────────────────────────────┘
```
**Giá trị:** Features là nền tảng. Không có features tốt → model kém → tools không giúp gì.

### Phase 2: Backtesting → Đánh giá Features + Model
```
Backtesting Framework ─────────────────────────────────────────
   │                                                          │
   │  Sharpe Ratio, Drawdown, Calmar = ĐÁNH GIÁ               │
   │  → Biết feature nào THỰC SỰ có giá trị                   │
   │  → Loại bỏ features noise, giữ features alpha            │
   │  → Model tối ưu hơn                                      │
   │                                                          │
   └──────────────────────────────────────────────────────────┘
```
**Giá trị:** Backtesting giúp validate — không phải features nào cũng có giá trị. Cần đo lường.

### Phase 3: Multi-Agent → Tổng hợp mọi thứ
```
Multi-Agent Architecture ──────────────────────────────────────
   │                                                          │
   │  Technical Agent  ← 42 features + Transformer + XGBoost  │
   │  Sentiment Agent  ← news + Polymarket odds (PMXT)        │
   │  Macro Agent      ← VIX, Bond, DXY features              │
   │  Risk Agent       ← MFI, volume, ATR features            │
   │  Bull/Bear Debate ← tổng hợp tất cả agents              │
   │  Fund Manager     ← ra quyết định cuối: BUY/SELL/HOLD   │
   │                                                          │
   │  = HỆ THỐNG HOÀN CHỈNH TỪ DATA → DECISION              │
   │                                                          │
   └──────────────────────────────────────────────────────────┘
```
**Giá trị:** Multi-agent kết hợp TẤT CẢ features + tools thành hệ thống ra quyết định thống nhất.

## C4. Kết luận — Tại sao cần CẢ Features LẪN Tools?

| Chỉ có Features, không có Tools | Chỉ có Tools, không có Features | Có CẢ HAI |
|---|---|---|
| Model dự đoán tốt nhưng không biết **khi nào nên tin** | Agent thông minh nhưng **không có dữ liệu chất lượng** | Model dự đoán tốt + Agent biết **đánh giá và ra quyết định** |
| Không đo được Sharpe/Drawdown → không biết risk | Garbage in = Garbage out | Backtesting → biết features nào tốt |
| Không phát hiện whale activity | Không có proxy cho institutional flow | Whale detection + proxy features = capture institutional signal |
| Predict = 1 con số | Predict = 1 con số + reasoning + confidence + risk assessment |

> **Tóm lại:** Features là "nguyên liệu thô", Tools là "đầu bếp và nhà hàng". Nguyên liệu tốt + đầu bếp giỏi = bữa ăn hoàn hảo. Thiếu một trong hai đều không đạt kết quả tối ưu.

---

## PHỤ LỤC — Công thức tính 8 Features mới

### MFI (Money Flow Index)
```
Typical Price = (High + Low + Close) / 3
Raw Money Flow = Typical Price × Volume
Positive MF = Σ(Raw MF khi TP tăng) trong 14 ngày
Negative MF = Σ(Raw MF khi TP giảm) trong 14 ngày
Money Ratio = Positive MF / Negative MF
MFI = 100 - 100/(1 + Money Ratio)
```

### Dividend Features
```
dividend_flag = 1 nếu date == exright_date, 0 nếu không
days_to_dividend = min(trading_days_until_next_exright, 60)
days_after_dividend = min(trading_days_since_last_exright, 60)
```

### Foreign Net Buy Proxy
```
CLV = (2×Close - High - Low) / (High - Low + 1e-9)
volume_zscore = (Volume - Mean_20) / (Std_20 + 1e-9)
foreign_net_buy_proxy = volume_zscore × CLV
foreign_net_buy_5d = rolling_mean(proxy, 5)
foreign_net_buy_20d = rolling_mean(proxy, 20)
```

### Self Net Buy Proxy
```
mfi_change = MFI(t) - MFI(t-1)
price_change = Close(t) / Close(t-1) - 1
mfi_divergence = mfi_change × sign(-price_change)  # Divergence signal
large_vol = 1 if volume_zscore > 2.0 else 0
self_net_buy_proxy = mfi_divergence × (1 + large_vol)
```
