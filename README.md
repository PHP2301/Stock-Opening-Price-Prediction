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
6. [Chi tiết các đặc trưng (34 Features)](#6-chi-tiết-các-đặc-trưng-34-features)
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

Mô hình sử dụng chuỗi **45 phiên giao dịch liên tiếp** làm đầu vào. Đầu vào Deep Learning có dạng tensor 3 chiều: `(N, 45, 34)` — cho phép Transformer tìm kiếm mối tương quan tuần hoàn và xu hướng trung hạn.

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
│   ├── run_pipeline.py         # Pipeline đầy đủ (Huấn luyện -> Dự báo)
│   ├── run_training.py         # Huấn luyện mô hình Hybrid Stacking
│   ├── run_backtest.py         # Mô phỏng giao dịch thực tế
│   └── run_tuning.py           # Tối ưu siêu tham số bằng Optuna
├── src/                        # Mã nguồn cốt lõi
│   ├── data_loader.py          # Tải dữ liệu Yahoo/DNSE, lọc nhiễu tỷ giá, đồng bộ timezone
│   ├── features.py             # Tính toán 34 chỉ báo, bộ lọc Kalman, trích xuất window
│   ├── ai_models.py            # Kiến trúc XGBoost & Conv1D-Transformer (Multi-task)
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
    D --> E[Timezone Sync: shift(1) dữ liệu Mỹ cho mã Việt Nam]
    E --> F[Feature Engineering: 34 đặc trưng kỹ thuật, vĩ mô & lịch]
    F --> G[Lọc nhiễu Kalman Filter]
    G --> H[StandardScaler + Sliding Window 45 ngày]
    H --> I{Huấn luyện AI đa nhiệm}
    I --> J[XGBoost: RandomizedSearchCV + TimeSeriesSplit]
    I --> K[Conv1D-Transformer: Uncertainty Weighting + Huber Loss]
    J --> L[Giải mã Target Scaler]
    K --> L
    L --> M[Quản trị rủi ro: ATR Safety Band]
    M --> N[Kết quả dự báo + Khoảng an toàn]
```

---

## 6. Chi tiết các đặc trưng (34 Features)

Đặc trưng đầu vào được chia tách thành **3 nhánh xử lý độc lập** trong kiến trúc mạng nơ-ron:

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

---

## 7. Kiến trúc mô hình AI đa nhiệm

Kiến trúc **Conv1D-Transformer Phân Nhánh** kết hợp **XGBoost Stacking**:

1.  **Sơ đồ Phân Nhánh (Branched Input):**
    *   3 nhánh đặc trưng đi vào các lớp Conv1D độc lập để lọc nhiễu cục bộ và trích xuất đặc trưng không gian riêng biệt.
    *   Nhúng nhãn thời gian qua Positional Embedding.
2.  **Lớp Time-Decay Attention:**
    *   Tự động tính toán tầm quan trọng của các phiên lịch sử dựa trên khoảng cách thời gian (ngày gần hơn có trọng số lớn hơn).
3.  **Tối ưu hóa đa nhiệm (Multi-task Learning):**
    *   Hai output heads dự báo đồng thời `target_return` và `target_spread`.
    *   Sử dụng `UncertaintyWeightsLayer` để tự động cân bằng Huber Loss của 2 đầu ra trong quá trình backpropagation.
4.  **XGBoost Stacking:**
    *   Lấy vector embedding 32 chiều cuối cùng từ Transformer, kết hợp với đặc trưng thô ngày hiện tại để làm đầu vào cho XGBoost.

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

#### 3. Meta Platforms (META) - 669 phiên
*   **Tổng lợi nhuận chiến lược:** **+18.70%** (so với Buy & Hold: **-19.50%**)
*   **Tỷ lệ Sharpe:** **+1.12**
*   **Mức rút vốn lớn nhất (MDD):** **-5.10%** (so với Buy & Hold: **-38.40%**)
*   **Số lệnh phát sinh:** 42 lệnh

---

## 9. Quản trị rủi ro & Dải bảo vệ ATR

Hệ thống cung cấp dải dự báo an toàn dựa trên độ biến động thị trường:

$$\text{Khoảng giá an toàn} = \text{Giá dự báo} \pm 1.5 \times \text{ATR}_{14}$$

Phân loại mức độ rủi ro dựa trên độ rộng dải an toàn để hạn chế giao dịch vào những ngày thị trường biến động cực đoan.

---

## 10. Xử lý lỗi dữ liệu tỷ giá USD/VND

Dữ liệu tỷ giá thô từ Yahoo Finance thường xuất hiện lỗi đột biến (outliers) hạ thấp tỷ giá đi hàng nghìn lần (ví dụ từ 23,000 xuống 23.00). Hệ thống loại bỏ hoàn toàn nhiễu qua bộ lọc 3 lớp:
1.  Nhân tỷ giá nhỏ hơn 1,000 với 1,000.
2.  Gán `NaN` cho các tỷ giá nằm ngoài biên độ thực tế `[15000, 28000]`.
3.  Lấp đầy giá trị thiếu bằng phương pháp Forward Fill và Backward Fill.

---

## 11. Kết quả đánh giá mô hình

Đo lường sai số dự báo MAE và MAPE trên tập kiểm thử độc lập:

### Vinamilk (VNM.VN)
*   **XGBoost:** MAE: 228.09 VND | MAPE: 0.38%
*   **Transformer:** MAE: 218.96 VND | MAPE: 0.36%

### Alphabet (GOOGL)
*   **XGBoost:** MAE: ~$1.60 | MAPE: 0.84%
*   **Transformer:** MAE: ~$1.70 | MAPE: 0.89%

### Meta Platforms (META)
*   **XGBoost:** MAE: ~$7.52 | MAPE: 1.31%
*   **Transformer:** MAE: ~$6.16 | MAPE: 1.08%

---

## 12. Hướng dẫn cài đặt và chạy (Local & Docker)

### Chạy trực tiếp (Local)

1.  **Cài đặt môi trường:**
    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```
2.  **Chạy toàn bộ Pipeline Huấn luyện & Dự báo:**
    ```powershell
    python scripts/run_pipeline.py
    ```
3.  **Chạy Backtest giao dịch:**
    ```powershell
    python scripts/run_backtest.py VNM.VN 0.0010
    ```
4.  **Khởi chạy Giao diện Web API:**
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
