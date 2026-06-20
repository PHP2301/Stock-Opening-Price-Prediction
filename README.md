# Stock Opening Price Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.21+](https://img.shields.io/badge/tensorflow-2.21+-orange.svg)](https://tensorflow.org/)
[![XGBoost 3.2+](https://img.shields.io/badge/xgboost-3.2+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hệ thống dự báo tỷ suất sinh lời giá mở cửa phiên giao dịch tiếp theo cho các mã chứng khoán **VNM.VN** (Vinamilk), **GOOGL** (Alphabet/Google) và **META** (Meta Platforms), sử dụng kiến trúc lai ghép thông minh giữa XGBoost và Conv1D-Transformer đa nhiệm (Multi-task Learning).

Dự án được xây dựng phục vụ nghiên cứu định lượng (quantitative research), tập trung vào độ chính xác dự báo, đồng bộ múi giờ quốc tế chống rò rỉ thông tin (data leakage), quản trị rủi ro và khả năng deploy thực tế bằng Docker.

---

## Mục lục

1. [Nguyên lý thiết kế](#1-nguyên-lý-thiết-kế)
2. [Thuật ngữ & Khái niệm Nghiệp vụ](#2-thuật-ngữ--khái-niệm-nghiệp-vụ)
3. [Thuật ngữ Công nghệ & Trí tuệ Nhân tạo (AI)](#3-thuật-ngữ-công-nghệ--trí-tuệ-nhân-tạo-ai)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Luồng xử lý dữ liệu](#5-luồng-xử-lý-dữ-liệu)
6. [Chi tiết các đặc trưng (82 đặc trưng - 42 cơ bản & 40 latent)](#6-chi-tiết-các-đặc-trưng-82-đặc-trưng---42-cơ-bản--40-latent)
7. [Kiến trúc mô hình AI đa nhiệm](#7-kiến-trúc-mô-hình-ai-đa-nhiệm)
8. [Hệ thống kiểm thử Backtest & Dynamic Slippage](#8-hệ-thống-kiểm-thử-backtest--dynamic-slippage)
9. [Quản trị rủi ro & Dải bảo vệ ATR](#9-quản-trị-rủi-ro--dải-bảo-vệ-atr)
10. [Xử lý lỗi dữ liệu tỷ giá USD/VND](#10-xử-lý-lỗi-dữ-liệu-tỷ-giá-usdvnd)
11. [Kết quả đánh giá mô hình](#11-kết-quả-đánh-giá-mô-hình)
12. [Hướng dẫn cài đặt và chạy (Local & Docker)](#12-hướng-dẫn-cài-đặt-và-chạy-local--docker)

---

## 1. Nguyên lý thiết kế

### Dự đoán tỷ suất sinh lời thay vì giá tuyệt đối

Giá cổ phiếu là chuỗi thời gian không dừng (non-stationary). Dự đoán trực tiếp giá tuyệt đối thường dẫn đến hiện tượng mô hình bị trễ (lagging), chỉ sao chép lại giá của ngày hôm trước. Thay vào đó, hệ thống dự đoán **tỷ suất sinh lời mở cửa (Opening Return)** và **độ rộng chênh lệch giá mở so với đóng hôm trước (Opening Spread)**:

$$\text{target-return} = \frac{\text{Open}_{T} - \text{Close}_{T-1}}{\text{Close}_{T-1}}$$

Sau khi có tỷ suất dự đoán, giá mở cửa được giải mã:

$$\text{Predicted Open} = \text{Close}_{today} \times (1 + \text{target-return}_{predicted})$$

### Cửa sổ trượt (Lookback Window)

Mô hình sử dụng chuỗi **45 phiên giao dịch liên tiếp** làm đầu vào. Đầu vào Deep Learning có dạng tensor 3 chiều: `(N, 45, 42)` — cho phép Transformer tìm kiếm mối tương quan tuần hoàn và xu hướng trung hạn.

### Chuẩn hóa StandardScaler

Hệ thống sử dụng StandardScaler (mean=0, std=1) thay cho MinMaxScaler. StandardScaler giữ nguyên tỷ lệ biến động thực tế, ít nhạy cảm với các nhiễu biên độ đột biến, và đẩy nhanh tốc độ hội tụ khi huấn luyện Gradient Descent.

---

## 2. Thuật ngữ & Khái niệm Nghiệp vụ

*   **Phiên ATO (At the Open):** Phiên khớp lệnh định kỳ xác định giá mở cửa lúc bắt đầu ngày giao dịch (tại Việt Nam diễn ra từ 9h00–9h15).
*   **Tín hiệu đồng thuận (Consensus Signal):** Điều kiện kích hoạt giao dịch mua/bán khi cả hai mô hình (XGBoost Lai & Transformer) đồng thời báo giá mở cửa ngày mai tăng vượt mức ngưỡng an toàn (ví dụ: > +0.10%).
*   **Slippage (Trượt giá):** Khoảng chênh lệch giữa giá đóng cửa ngày hôm trước và giá khớp lệnh thực tế do độ trễ truyền dữ liệu hoặc thanh khoản mỏng.
*   **Trượt giá động (Dynamic Slippage):** Phí trượt giá tự động tăng gấp đôi nếu thanh khoản thị trường tại ngày hôm đó rơi vào trạng thái cạn kiệt (Z-score khối lượng < -1.0).

---

## 3. Thuật ngữ Công nghệ & Trí tuệ Nhân tạo (AI)

*   **GLU (Gated Linear Unit):** Cổng tuyến tính có cổng chặn giúp tự động sàng lọc thông tin quan trọng trước khi đưa vào các lớp tiếp theo.
*   **Self-Attention & Time-Decay Attention:** Cơ chế tự chú ý kết hợp hàm suy giảm theo thời gian, giúp mô hình ưu tiên các thông tin gần ngày hiện tại hơn nhưng vẫn giữ kết nối với quá khứ.
*   **Uncertainty Weighting Loss:** Phương pháp tự động tối ưu hóa trọng số tổn thất đa nhiệm (Huber Loss của Return và Spread) dựa trên mức độ không chắc chắn tự động học được của từng tác vụ.
*   **Keras 3 Custom Serialization:** Giải pháp tách các tham số học tập tự do (`log_var`) vào một Layer độc lập (`UncertaintyWeightsLayer`) nằm bên trong Graph mô hình, loại bỏ triệt để lỗi revive/deserialization khi lưu và tải model dưới dạng Functional Model trong Keras 3.

---

## 4. Cấu trúc thư mục

```
Stock-Opening-Price-Prediction/
├── config/                     # Cấu hình siêu tham số Optuna (.json)
├── data/                       # Dữ liệu giá thô và đặc trưng đã xử lý (.csv)
├── docs/                       # Tài liệu nghiên cứu, tasklist và implementation plan
├── logs/                       # Lịch sử dự báo và log hệ thống
├── models/                     # Mô hình đã huấn luyện (.pkl, .keras) và scalers
├── notebooks/                  # Jupyter Notebooks phân tích từng module
├── reports/
│   └── figures/                # Biểu đồ so sánh dự báo và backtest equity curve
├── scripts/                    # Các script thực thi chính
│   ├── run_training_transformer.py # Huấn luyện mô hình Hybrid Stacking (Transformer + XGBoost)
│   ├── run_backtest.py         # Mô phỏng giao dịch thực tế & kiểm tra Walk-Forward
│   ├── run_tuning.py           # Tối ưu siêu tham số bằng Optuna độc lập cho từng mã
│   └── predict.py              # Đưa ra dự báo hàng ngày & kích hoạt Multi-Agent Desk
├── src/                        # Mã nguồn cốt lõi
│   ├── data_loader.py          # Tải dữ liệu Yahoo/DNSE, lọc nhiễu tỷ giá, đồng bộ timezone
│   ├── features.py             # Tính toán 42 chỉ báo, bộ lọc Kalman, trích xuất window
│   ├── ai_models.py            # Kiến trúc XGBoost & Conv1D-Transformer đa nhiệm (Multi-task)
│   ├── agents/                 # PydanticAI Multi-Agent debate system (Technical, Macro, Sentiment, Risk)
│   └── web/                    # FastAPI Backend và HTML/CSS/JS Frontend
├── tests/                      # Bộ unit tests tự động
├── Dockerfile                  # Cấu hình đóng gói container Docker
└── docker-compose.yml          # Triển khai container hóa hệ thống web app
```

---

## 5. Luồng xử lý dữ liệu

```mermaid
graph TD
    A[Yahoo Finance + DNSE API] --> B[Tải tỷ giá USD/VND trực tuyến]
    B --> C[Bộ lọc tỷ giá 3 lớp]
    C --> D[Quy đổi USD sang VND theo tỷ giá động]
    D --> E["Timezone Sync: shift(1) dữ liệu Mỹ cho mã Việt Nam"]
    E --> F[Feature Engineering: 42 đặc trưng cơ bản + 40 đặc trưng biểu diễn ẩn = Tổng 82 đặc trưng]
    F --> G[Lọc nhiễu Kalman Filter]
    G --> H[StandardScaler + Sliding Window 45 ngày]
    H --> I{Huấn luyện AI đa nhiệm}
    I --> J[XGBoost: GridSearchCV + TimeSeriesSplit]
    I --> K[Conv1D-Transformer: Uncertainty Weighting + Huber Loss]
    J --> L[Giải mã Target Scaler]
    K --> L
    L --> M[Inference: predict.py]
    M --> N[Multi-Agent Desk: PydanticAI debate]
    N --> O[Quản trị rủi ro: ATR Safety Band & Position Sizing]
    O --> P[Kết quả quyết định & Báo cáo chi tiết]
```

---

## 6. Chi tiết các đặc trưng (82 đặc trưng - 42 cơ bản & 40 latent)

Đặc trưng đầu vào được chia tách thành **4 nhánh xử lý độc lập** trong kiến trúc mạng nơ-ron:

### Nhánh 1: Giá & Động lượng (12 đặc trưng)
*   `gap_open`: Chênh lệch giá mở cửa hôm nay so với đóng cửa hôm trước.
*   `open_return`: Tỷ suất sinh lời mở cửa.
*   `buying_pressure`: Áp lực mua trong phiên $((Close - Low) / (High - Low + 1e-9))$.
*   `shadow_ratio`: Tỷ lệ bóng nến trên/dưới $((High - Close) / (Close - Low + 1e-9))$.
*   `intraday_range`: Biên độ dao động giá nội phiên.
*   `return_1d`, `return_2d`, `return_3d`: Tỷ suất sinh lời đóng cửa các phiên trước.
*   `mom_5d`, `mom_10d`, `mom_20d`: Chỉ báo động lượng động thái giá.
*   `dist_ma50`: Khoảng cách tương đối từ giá hiện tại đến đường SMA 50.

### Nhánh 2: Khối lượng & Biến động (6 đặc trưng)
*   `volume_change`: Tốc độ thay đổi khối lượng giao dịch.
*   `volume_sma_ratio`: Khối lượng hiện tại so với trung bình 20 phiên.
*   `volume_zscore`: Điểm chuẩn hóa Z-score khối lượng (để kích hoạt slippage động).
*   `ad_line_ratio`: Chỉ báo tích lũy/phân phối chuẩn hóa $((Close - Low - (High - Close)) / (High - Low + 1e-9))$.
*   `obv_zscore`: Z-score của chỉ số khối lượng cân bằng tích lũy OBV.
*   `vol_ratio`: Biến động khối lượng tương đối.

### Nhánh 3: Kỹ thuật, Vĩ mô & Lịch (16 đặc trưng)
*   `rsi_14`: Chỉ số sức mạnh tương đối.
*   `macd_ratio`: Tỷ lệ đường MACD trên đường tín hiệu Signal.
*   `bb_position`: Vị trí tương đối của giá trong dải Bollinger Bands.
*   `adx_14`: Chỉ số định hướng trung bình (đo sức mạnh xu hướng).
*   `stoch_k`: Chỉ báo dao động ngẫu nhiên Stochastic %K.
*   `efficiency_ratio`: Chỉ số hiệu quả Kaufman (đo lường độ nhiễu của giá).
*   `vix_lag1`: Chỉ số biến động VIX trễ 1 ngày (Mỹ).
*   `bond_yield_lag1`: Lợi suất trái phiếu chính phủ Mỹ 10 năm trễ 1 ngày.
*   `usdvnd_change`: Biến động tỷ giá USD/VND.
*   `vnindex_return_lag1`: Lợi suất chỉ số VN-Index (đối với mã Việt Nam).
*   `day_of_week_sin` / `cos`: Mã hóa dạng sóng tuần hoàn cho ngày trong tuần.
*   `month_sin` / `cos`: Mã hóa dạng sóng tuần hoàn cho tháng trong năm.
*   `is_quarter_end`: Đánh dấu các ngày chốt sổ cuối quý.
*   `days_before_tet`: Số ngày đếm ngược đến Tết Nguyên Đán (áp dụng riêng cho VNM.VN).

### Nhánh 4: Dòng tiền & Cổ tức (8 đặc trưng mới)
*   `mfi_14`: Chỉ số dòng tiền Money Flow Index.
*   `dividend_flag`: Tín hiệu ngày không hưởng quyền (ex-dividend date).
*   `days_to_dividend`: Số ngày đếm ngược đến kỳ cổ tức tiếp theo.
*   `days_after_dividend`: Số ngày đã trôi qua kể từ kỳ cổ tức gần nhất.
*   `foreign_net_buy_proxy`: Chỉ số proxy giao dịch khối ngoại dựa trên Z-score volume và Close Location Value (CLV).
*   `foreign_net_buy_5d`: Proxy ngoại trung bình động 5 ngày.
*   `foreign_net_buy_20d`: Proxy ngoại trung bình động 20 ngày.
*   `self_net_buy_proxy`: Chỉ số proxy giao dịch tự doanh dựa trên phân kỳ MFI và động lượng giá.

---

## 7. Kiến trúc mô hình AI đa nhiệm

Kiến trúc **Conv1D-Transformer Phân Nhánh** kết hợp **XGBoost Stacking**:

1.  **Sơ đồ Phân Nhánh (Branched Input):**
    *   4 nhánh đặc trưng (Giá & Động lượng, Khối lượng & Biến động, Kỹ thuật/Vĩ mô/Lịch, Dòng tiền & Cổ tức) đi vào các lớp Conv1D độc lập để trích xuất đặc trưng không gian riêng biệt.
    *   Đưa qua lớp nhúng vị trí thời gian Positional Embedding.
2.  **Lớp Time-Decay Attention & 4 Nhánh ẩn (Latent Branches):**
    *   Tự động tính toán tầm quan trọng của các phiên lịch sử dựa trên khoảng cách thời gian (ngày gần hơn có trọng số lớn hơn).
    *   Mỗi nhánh trích xuất đặc trưng latent riêng: Nhánh Giá (16 chiều), Nhánh Khối lượng (8 chiều), Nhánh Kỹ thuật (8 chiều), Nhánh Dòng tiền/Cổ tức (8 chiều). Tổng vector latent embedding = 40 chiều.
3.  **Tối ưu hóa đa nhiệm (Multi-task Learning & Uncertainty Weighting):**
    *   Hai output heads dự báo đồng thời `target_return` (tỷ suất lợi nhuận T+1, T+2, T+3) và `target_spread` (chênh lệch giá mở/đóng T+1, T+2, T+3).
    *   Sử dụng `UncertaintyWeightsLayer` để tự động cân bằng Huber Loss của 2 đầu ra (Return và Spread) dựa trên mức độ không chắc chắn tự động học được.
4.  **XGBoost Stacking:**
    *   Lấy vector embedding 40 chiều cuối cùng từ Transformer, kết hợp với 42 đặc trưng thô của ngày hiện tại để tạo thành vector đầu vào **82 chiều** cho mô hình XGBoostRegressor cực kỳ mạnh mẽ.

---

## 8. Hệ thống kiểm thử Backtest & Dynamic Slippage

Giao dịch giả lập Overnight Trading out-of-sample (2023–2026):

*   **Quy tắc:** Mua tại Close ngày $T-1$ nếu cả hai mô hình đồng thuận báo tăng vượt ngưỡng kích hoạt $Th$ (ví dụ: $+0.10\%$). Bán tại Open ngày $T$.
*   **Dynamic Slippage:** Tự động tăng gấp đôi phí trượt giá cơ sở nếu thanh khoản thị trường tại ngày đó cực kỳ mỏng nhằm kiểm soát rủi ro khớp lệnh không thuận lợi.

### Kết quả Backtest Chi tiết (2023–2026, Ngưỡng $Th = +0.10\%$)

#### 1. Vinamilk (VNM.VN) - 693 phiên
*   **Tổng lợi nhuận chiến lược:** **+4.15%** (so với Buy & Hold: **-5.97%**)
*   **Tỷ lệ Sharpe:** **+0.48**
*   **Mức rút vốn lớn nhất (MDD):** **-2.10%** (so với Buy & Hold: **-31.70%**)
*   **Số lệnh phát sinh:** 24 lệnh

#### 2. Alphabet (GOOGL) - 669 phiên
*   **Tổng lợi nhuận chiến lược:** **+12.45%** (so với Buy & Hold: **-14.20%**)
*   **Tỷ lệ Sharpe:** **+0.85**
*   **Mức rút vốn lớn nhất (MDD):** **-4.85%** (so với Buy & Hold: **-28.90%**)
*   **Số lệnh phát sinh:** 38 lệnh

#### 3. Meta Platforms (META) - 650 phiên (Cập nhật sau khi nâng cấp hệ 82 đặc trưng)
*   **Tổng lợi nhuận chiến lược:** **+8.27%** (so với Buy & Hold: **+107.28%**)
*   **Tỷ lệ Sharpe:** **+0.53**
*   **Mức rút vốn lớn nhất (MDD):** **-4.25%** (so với Buy & Hold: **-33.14%**)
*   **Số lệnh phát sinh:** 9 lệnh
*   **Tỷ lệ thắng (Win Rate):** **55.56%**
*   **Profit Factor:** **2.05**
*   **Chi tiết Walk-Forward Validation:**
    *   **Window 1 (2023-10-11 → 2024-08-20):** Chiến lược: **+0.20%** | B&H: **+68.64%** | Sharpe: **0.07** | MDD: **-3.98%**
    *   **Window 2 (2024-08-21 → 2025-07-02):** Chiến lược: **+11.24%** | B&H: **+43.37%** | Sharpe: **1.50** | MDD: **-4.25%**
    *   **Window 3 (2025-07-03 → 2026-05-14):** Chiến lược: **-2.86%** | B&H: **-12.83%** | Sharpe: **-0.70** | MDD: **-3.41%**
    *(Chiến lược bảo toàn vốn cực kỳ tốt trong giai đoạn thị trường Downtrend của Window 3: chỉ giảm -2.86% so với mức giảm -12.83% của B&H).*

## 9. Quản trị rủi ro, Dải bảo vệ ATR & Tối ưu vị thế Kelly Criterion

Hệ thống tích hợp quy trình kiểm soát rủi ro đa lớp:

### A. Dải bảo vệ ATR (Average True Range)
Cung cấp dải dự báo an toàn động dựa trên độ biến động thực tế của thị trường:

$$\text{Khoảng giá an toàn} = \text{Giá dự báo} \pm 1.5 \times \text{ATR}_{14}$$

Nếu biên độ dao động dự kiến vượt quá dải ATR hoặc ATR ngày hôm đó tăng vọt bất thường ($\ge 3\%$), hệ thống sẽ tự động phân loại giao dịch là rủi ro cao và đưa ra khuyến nghị hạn chế giao dịch.

### B. Công thức tối ưu hóa vị thế Kelly Criterion
Để quản trị quy mô lệnh (Position Sizing), `RiskAgent` áp dụng công thức **Half-Kelly** có kiểm soát rủi ro:

1. **Xác suất thắng thực tế $p$**: Tránh Overconfidence của AI bằng cách lấy giá trị nhỏ nhất giữa độ tự tin tranh luận của Orchestrator (`confidence_score`) và tỷ lệ thắng lịch sử (`win_rate_history` lấy từ Backtest):
   $$p = \min(\text{confidence\_score}, \text{win\_rate\_history})$$
2. **Tỷ lệ vị thế (Position Size) $f^*$**:
   $$f^* = 0.5 \times \left( p - \frac{1 - p}{b} \right)$$
   Trong đó:
   * $b$: Tỷ lệ Risk-to-Reward (Reward / Risk) được tính từ khoảng cách Take Profit / Stop Loss.
   * Hệ số $0.5$ (Half-Kelly) được sử dụng để giảm thiểu biến động tài sản lớn (Drawdown).
   * **Hard Cap**: Tỷ lệ phân bổ vốn tối đa cho một lệnh được giới hạn cứng ở mức **25%** tổng tài sản.

---

## 10. Hệ thống Đa Agent PydanticAI (Multi-Agent Debate Desk)

Để nâng cấp dự báo thô của AI lên quyết định giao dịch thực tế, dự án tích hợp hệ thống đa Agent được điều phối qua **PydanticAI**:

*   **TechnicalAgent:** Phân tích các đặc trưng kỹ thuật, tín hiệu dự báo thô từ Transformer và XGBoost, các ngưỡng SMA, RSI, MFI.
*   **SentimentAgent:** Phân tích điểm số cảm xúc tin tức thu thập thời gian thực qua nguồn RSS Feed chính thống (CafeF RSS cho cổ phiếu Việt Nam, Google News RSS cho cổ phiếu Mỹ) để loại bỏ nhiễu và cào tin tức an toàn/hợp lệ, kết hợp tích hợp các tỷ lệ odds của thị trường dự đoán Polymarket (PMXT) cho các cổ phiếu Mỹ.
*   **MacroAgent:** Đánh giá các biến số vĩ mô như Bond Yield, VIX, và Dollar Index để xác định độ an toàn của dòng tiền liên thị trường.
*   **RiskAgent:** Tính toán mức độ rủi ro dựa trên ATR và xác định tỷ lệ phân bổ vốn (Position Sizing) tối ưu.
*   **Orchestrator Agent:** Sử dụng PydanticAI để tổ chức tranh luận (Bull vs Bear Debate) giữa các agent, sau đó trả về quyết định giao dịch cuối cùng cấu trúc hóa dạng JSON (Action: Buy/Sell/Hold, Confidence Score, Stop Loss, Take Profit, and Reasoning).

---

## 11. Xử lý lỗi dữ liệu tỷ giá USD/VND

Dữ liệu tỷ giá thô từ Yahoo Finance thường xuất hiện lỗi đột biến (outliers) hạ thấp tỷ giá đi hàng nghìn lần (ví dụ từ 23,000 xuống 23.00). Hệ thống loại bỏ hoàn toàn nhiễu qua bộ lọc 3 lớp:
1.  Nhân tỷ giá nhỏ hơn 1,000 với 1,000.
2.  Gán `NaN` cho các tỷ giá nằm ngoài biên độ thực tế `[15000, 28000]`.
3.  Lấp đầy giá trị thiếu bằng phương pháp Forward Fill và Backward Fill.

---

## 12. Kết quả đánh giá mô hình

Đo lường sai số dự báo MAE và MAPE trên tập kiểm thử độc lập (đã cập nhật hệ 82 đặc trưng):

### Vinamilk (VNM.VN)
*   **XGBoost:** MAE: 247.45 VND | MAPE: 0.41%
*   **Transformer:** MAE: 221.68 VND | MAPE: 0.37%

### Alphabet (GOOGL)
*   **XGBoost:** MAE: ~$3.32 (87.353 VND) | MAPE: 1.79%
*   **Transformer:** MAE: ~$1.58 (41.461 VND) | MAPE: 0.85%

### Meta Platforms (META)
*   **XGBoost:** MAE: ~$6.72 (176.485 VND) | MAPE: 1.23%
*   **Transformer:** MAE: ~$5.16 (135.556 VND) | MAPE: 0.96%

---

## 13. Hướng dẫn cài đặt và chạy (Local & Docker)

### Chạy trực tiếp (Local)

1.  **Cài đặt môi trường:**
    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```
2.  **Tối ưu siêu tham số (Optuna):**
    ```powershell
    python scripts/run_tuning.py META
    ```
3.  **Huấn luyện mô hình cuộn chiếu & sản xuất:**
    *(Lưu ý đối với hệ điều hành Windows: Cần thiết lập biến môi trường TensorFlow để tránh lỗi)*
    ```powershell
    $env:TF_ENABLE_ONEDNN_OPTS=0
    python scripts/run_training_transformer.py META
    ```
4.  **Chạy dự báo hàng ngày & Multi-Agent Debate Desk:**
    ```powershell
    python scripts/predict.py META
    ```
5.  **Dọn dẹp các tệp mô hình cũ (Giảm dung lượng ổ đĩa):**
    ```powershell
    python scripts/clean_old_models.py
    ```
6.  **Chạy Backtest giao dịch:**
    ```powershell
    python scripts/run_backtest.py META 0.0010
    ```
7.  **Khởi chạy Giao diện Web API:**
    ```powershell
    python src/web_runner/run_web.py
    ```
    Truy cập giao diện tại: `http://127.0.0.1:8000`

### Chạy bằng Docker Compose (Khuyên dùng)

Hệ thống hỗ trợ container hóa hoàn toàn ứng dụng web. Chạy lệnh sau để tự động tải thư viện, cấu hình môi trường, huấn luyện mô hình và mở cổng dịch vụ web:

```powershell
docker-compose up --build
```
Dịch vụ web FastAPI + Giao diện Frontend sẽ tự động khởi chạy tại: `http://localhost:8000`

---

## License

Dự án được phân phối dưới giấy phép MIT License. Xem file [LICENSE](LICENSE) để biết thêm thông tin chi tiết.
