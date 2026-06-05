# 📑 HƯỚNG DẪN & GHI CHÚ HỆ THỐNG DỰ BÁO GIÁ MỞ CỬA

Tài liệu này ghi nhận toàn bộ cấu trúc dữ liệu, phương pháp huấn luyện, và chi tiết các mô hình AI đang được sử dụng trong dự án nhằm giúp bạn dễ dàng theo dõi và báo cáo.

---

## 1. DỮ LIỆU ĐẦU VÀO (DATA PIPELINE)

### 📊 Nguồn dữ liệu & Đồng nhất đơn vị

Hệ thống sử dụng cơ chế gộp dữ liệu thông minh từ 2 nguồn chính cho mỗi mã cổ phiếu:

1.  **Dữ liệu Vinamilk (VNM.VN):**
    - **Dữ liệu trường cấp (`data/VNM_prices.csv`):** Từ 17/09/2019 đến 16/03/2026. Đơn vị gốc: Nghìn VNĐ. Hệ thống tự động nhân với 1000 để quy đổi về VNĐ.
    - **Dữ liệu lịch sử bổ sung (DNSE API):** Tải tự động từ năm 2012 giúp mở rộng tập dữ liệu lên **3.535 phiên giao dịch** (tăng thêm 1.868 phiên).
2.  **Dữ liệu các mã công nghệ Mỹ (GOOGL & META):**
    - Tải trực tiếp từ **Yahoo Finance API** bắt đầu từ mốc **2010-01-01** (GOOGL) và **2012-05-18** (ngày META IPO).
    - Đồng nhất đơn vị tiền tệ: Quy đổi toàn bộ giá trị USD sang **VNĐ** theo tỷ giá trực tuyến thời gian thực (realtime exchange rate).
    - Tổng hợp dữ liệu: **4.119 phiên** (GOOGL) và **3.520 phiên** (META).

---

### 📈 Các đặc trưng đầu vào (24 Features)

Để dự báo xu hướng, hệ thống tính toán 24 chỉ báo kỹ thuật và vĩ mô dạng dừng (stationary features) sau khi làm mịn qua bộ lọc nhiễu **Kalman Filter**:

- `close_smoothed`: Giá đóng cửa được làm mịn bằng Kalman Filter để khử nhiễu ngắn hạn.
- `close_lag1`: Giá đóng cửa của ngày hôm trước (độ trễ 1 phiên).
- `volume_change`: Tỷ lệ thay đổi khối lượng giao dịch so với phiên trước.
- `intraday_return`: Tỷ suất sinh lời trong ngày `(Close - Open) / Open`.
- `volatility_20`: Độ lệch chuẩn của tỷ suất sinh lời trong 20 phiên gần nhất (đo lường biến động).
- `rsi_14`: Chỉ số sức mạnh tương đối (Relative Strength Index).
- `macd_ratio`: Chỉ báo MACD line được chia cho Close hiện tại để dừng hóa đặc trưng.
- `bb_lower`, `bb_middle`, `bb_upper`: Dải dưới, dải giữa, dải trên của Bollinger Bands (chu kỳ 20 phiên, độ lệch 2) dạng tỷ lệ so với Close.
- `atr_ratio`: Chỉ báo biên độ dao động thực tế trung bình (Average True Range) chia cho Close.
- `ema_14_ratio`: Đường trung bình di động lũy thừa (EMA 14 phiên) dạng tỷ lệ so với Close.
- `roc_10`: Tốc độ thay đổi giá (Rate of Change 10 phiên) đo lường động lượng giá.
- `adx_14`: Chỉ số định hướng trung bình (ADX 14 phiên) xác định cường độ mạnh/yếu của xu hướng hiện tại.
- `bond_yield_10y`: Lợi suất trái phiếu chính phủ Mỹ 10 năm (`^TNX`) đo lường biến động vĩ mô và dòng tiền liên thị trường.
- `dollar_index_change`: Biến động tỷ lệ ngày của chỉ số sức mạnh Dollar Index (`DX-Y.NYB`).

---

### 🎯 Biến mục tiêu dự báo (Target)

Mô hình dự báo **Tỷ suất lợi nhuận mở cửa ngày mai (`target_return`)**:
$$\text{target-return} = \frac{\text{Open}_{tomorrow} - \text{Close}_{today}}{\text{Close}_{today}}$$

Sau khi AI dự báo ra tỷ suất này, hệ thống sẽ tự động quy đổi ngược về giá tiền thực tế:
$$\text{Giá mở cửa dự báo} = \text{Close}_{today} \times (1 + \text{target-return}_{predicted})$$

---

## 2. PHƯƠNG PHÁP HUẤN LUYỆN (TRAINING FLOW)

- **Tính tái lập (Reproducibility):** Cố định seed toàn cục `SEED = 42` cho Numpy, Random, TensorFlow.
- **Cửa sổ trượt (Sliding Window):** Sử dụng `time_steps = 45` phiên giao dịch liên tiếp trong quá khứ (~2.5 tháng) làm chuỗi dữ liệu đầu vào.
- **Chuẩn hóa dữ liệu (Scaling):** Sử dụng `MinMaxScaler` đưa dữ liệu về khoảng `[0, 1]`. Bộ chuẩn hóa đầu vào (`feature_scaler`) và đầu ra (`target_scaler`) được tách riêng biệt.
- **Chia tập dữ liệu (Split Strategy):** Chia tỷ lệ **80% huấn luyện (Train) / 20% kiểm thử (Test)** theo trật tự thời gian (Chronological Split) để tránh rò rỉ dữ liệu tương lai.

---

## 3. CHI TIẾT CÁC MÔ HÌNH AI

Hệ thống huấn luyện song song hai mô hình tối ưu nhất:

### 1. 🌳 XGBoost (Extreme Gradient Boosting)

- **Đặc điểm:** Mô hình dạng cây quyết định nâng cao, rất mạnh với dữ liệu dạng bảng. Cần làm phẳng dữ liệu chuỗi 3D thành 2D trước khi đưa vào huấn luyện.
- **Tối ưu tham số (GridSearchCV):** Áp dụng quét lưới kết hợp kiểm thử chéo chuỗi thời gian (`TimeSeriesSplit` với 5 splits) chạy song song (`n_jobs=-1`) để tự động tìm kiếm bộ siêu tham số tốt nhất.
- **Bộ tham số tối ưu tìm được:**
  - **VNM.VN:** `{'subsample': 0.9, 'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.03, 'colsample_bytree': 1.0}`
  - **GOOGL:** `{'subsample': 0.8, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03, 'colsample_bytree': 0.8}`
  - **META:** `{'subsample': 0.8, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03, 'colsample_bytree': 0.8}`
- **Ý nghĩa chi tiết của các siêu tham số:**

| Siêu tham số                          | Ý nghĩa kỹ thuật                                                                       | Ý nghĩa thực tế trong dự án                                                                                                                                     |
| :------------------------------------ | :------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`n_estimators`** (150-200)           | Số lượng cây quyết định được tạo lập tuần tự để bổ trợ sai số cho nhau.                | Giúp mô hình đạt độ hội tụ sai số tối thiểu mà không tiêu tốn tài nguyên tính toán.                                                                             |
| **`learning_rate`** (0.03)            | Tốc độ học (hệ số co hẹp đóng góp của mỗi cây quyết định mới).                         | Giá trị nhỏ `0.03` giúp mô hình học từ từ qua từng cây, tránh hiện tượng nhảy quá đà (Overfitting) và khớp mịn hơn với xu hướng dài hạn.                        |
| **`max_depth`** (3 hoặc 4)            | Độ sâu tối đa (số tầng phân nhánh tối đa) của mỗi cây quyết định.                      | Cây nông ở mức 3 hoặc 4 giúp giới hạn độ phức tạp của mỗi cây, ngăn chặn việc cây học thuộc lòng các nhiễu nhỏ trong giá sàn.                                   |
| **`subsample`** (0.8 hoặc 0.9)        | Tỷ lệ số dòng dữ liệu (phiên giao dịch) được lấy mẫu ngẫu nhiên để huấn luyện mỗi cây. | Mỗi cây chỉ học trên 80-90% số phiên ngẫu nhiên. Việc này tạo ra sự đa dạng và giúp mô hình chống chịu tốt hơn trước các đột biến giá ngắn hạn (nhiễu thị trường). |
| **`colsample_bytree`** (0.8 hoặc 1.0) | Tỷ lệ số cột dữ liệu (đặc trưng kỹ thuật) được chọn ngẫu nhiên khi xây dựng mỗi cây.   | Rút ngẫu nhiên 80% hoặc dùng cả 100% số cột đặc trưng đầu vào giúp mô hình không bị lệ thuộc phiến diện vào một vài chỉ báo kỹ thuật cụ thể.                    |

### 2. 🤖 Transformer Encoder

- **Đặc điểm:** Kiến trúc Deep Learning tiên tiến nhất sử dụng cơ chế chú ý (Multi-Head Self-Attention) trực tiếp bắt trọn mối quan hệ phi tuyến phức tạp trong chuỗi thời gian 45 ngày.
- **Cấu trúc chi tiết:**
  - Lớp nhúng vị trí thời gian `PositionalEmbedding`.
  - 2 Khối Attention liên tiếp: Mỗi khối gồm **MultiHeadAttention (2 heads, key_dim=64)**, **Dropout (0.2582)**, Residual Connections và **Layer Normalization**.
  - 2 Lớp Feed-Forward Networks với 256 nơ-ron kích hoạt `ReLU`.
  - Lớp Flatten và các lớp Dense trung gian (128, 64, 32 nơ-ron) kết hợp Dropout giảm Overfitting.
  - Biên dịch với thuật toán tối ưu `Adam` với hệ số học nhỏ `5.117e-5` và hàm mất mát Huber.
  - Tích hợp bộ kiểm soát `EarlyStopping` (patience=25) và tự động giảm tốc độ học `ReduceLROnPlateau` để tránh overfitting.

---

## 4. KẾT QUẢ ĐÁNH GIÁ THỰC TẾ (SAU NÂNG CẤP ĐẶC TRƯNG DỪNG VÀ TỐI ƯU HÓA OPTUNA)

Dưới đây là sai số thực tế trên tập kiểm thử (Test Set) sau khi áp dụng 24 đặc trưng dừng (stationary features), lọc nhiễu Kalman Filter, bổ sung biến vĩ mô và tự động tìm siêu tham số tối ưu bằng Optuna:

### 🇻🇳 Vinamilk (VNM.VN) - Đơn vị: VNĐ

- 🤖 **Transformer gốc (Độc lập):**
  - _Sai số RMSE:_ **394,89 VNĐ**
  - _Sai số MAE:_ **221,68 VNĐ** (Lệch trung bình: **0.37%**) 🟢
- 🌳 **Hybrid XGBoost (Mô hình lai):**
  - _Sai số RMSE:_ **430,53 VNĐ**
  - _Sai số MAE:_ **247,45 VNĐ** (Lệch trung bình: **0.41%**) 🟢

### 🇺🇸 Alphabet / Google (GOOGL) - Đơn vị: VNĐ & USD

- 🤖 **Transformer gốc (Độc lập):**
  - _Sai số RMSE:_ **72.370,89 VNĐ** (~$2.75 USD)
  - _Sai số MAE:_ **41.461,79 VNĐ** (~$1.58 USD - Lệch trung bình: **0.85%**) 🟢
- 🌳 **Hybrid XGBoost (Mô hình lai):**
  - _Sai số RMSE:_ **118.370,28 VNĐ** (~$4.51 USD)
  - _Sai số MAE:_ **87.353,30 VNĐ** (~$3.32 USD - Lệch trung bình: **1.79%**) 🟢

### 🇺🇸 Meta Platforms (META) - Đơn vị: VNĐ & USD

- 🤖 **Transformer gốc (Độc lập):**
  - _Sai số RMSE:_ **249.928,68 VNĐ** (~$9.51 USD)
  - _Sai số MAE:_ **135.556,19 VNĐ** (~$5.16 USD - Lệch trung bình: **0.96%**) 🟢
- 🌳 **Hybrid XGBoost (Mô hình lai):**
  - _Sai số RMSE:_ **287.284,27 VNĐ** (~$10.93 USD)
  - _Sai số MAE:_ **176.485,17 VNĐ** (~$6.72 USD - Lệch trung bình: **1.23%**) 🟢


---

## 🛠️ HƯỚNG DẪN CHẠY PIPELINE TRÊN TERMINAL

Mỗi khi muốn chạy lại toàn bộ quá trình tải dữ liệu, tính toán đặc trưng, huấn luyện các mô hình và xuất dự báo cho ngày mai, bạn chỉ cần gõ lệnh:

```powershell
python scripts/run_pipeline.py
```

- **Kết quả đầu ra:** Các biểu đồ so sánh dự báo của các mô hình được lưu độc lập cho từng mã: `reports/figures/model_battle_result_VNM.VN.png`, `reports/figures/model_battle_result_GOOGL.png`, và `reports/figures/model_battle_result_META.png`.
- **Các mô hình đã huấn luyện** được tự động lưu trong thư mục `models/` để sử dụng dự báo nhanh mà không cần huấn luyện lại.
- **Để dọn dẹp các file rác phát sinh** (như Python cache, Jupyter checkpoints), bạn có thể chạy:

```powershell
python scripts/clean_workspace.py
```

---

## 5. THIẾT KẾ WEB APPLICATION (LỘ TRÌNH TƯƠNG LAI)

### 💾 Kế hoạch thiết kế Cơ sở dữ liệu (Database Design)
- **Môi trường Phát triển (Development):** Sử dụng **FastAPI + SQLite** để chạy thử nghiệm offline trên máy cá nhân. Cơ sở dữ liệu SQLite được lưu trữ dưới dạng một file duy nhất trong thư mục dự án (ví dụ: `data/processed/stock_predictions.db`).
- **Môi trường Triển khai (Production/Deploy):** Chuyển đổi sang sử dụng **PostgreSQL** chạy trên môi trường đám mây (như Neon, Supabase, Render) để hỗ trợ nhiều người dùng truy cập đồng thời và quản lý dữ liệu lớn ổn định hơn.

### 🔌 Vai trò của Backend FastAPI
- **Cầu nối dữ liệu:** Đọc/ghi dữ liệu từ Database và trả về dạng JSON chuẩn cho Giao diện người dùng (React/Next.js/HTML).
- **Tối ưu hóa mô hình AI:** Tải sẵn các mô hình XGBoost và Transformer lên RAM khi Server khởi động để thực hiện dự báo trong thời gian thực cực nhanh (vài ms).
- **Tác vụ nền (Background Tasks):** Tự động lập lịch chạy hàng ngày để cập nhật giá cổ phiếu, phân tích cảm xúc tin tức (NLP FinBERT) và lưu dự báo phiên kế tiếp vào database mà không làm gián đoạn người dùng.

---

## 6. MÔ HÌNH LAI (HYBRID MODEL) - CẬP NHẬT 29/05/2026 (LÀM SẠCH TIN TỨC & TÍCH HỢP CƠ SỞ DỮ LIỆU)

Hệ thống đã trải qua một đợt nâng cấp quan trọng liên quan đến dữ liệu tin tức cảm xúc và tối ưu hóa lõi AI:

### 📰 Làm sạch và Lọc tin tức phân tán theo mã cổ phiếu (Ticker-specific Filtering):
- **Loại bỏ nhiễu**: Loại bỏ hoàn toàn các tin tức không liên quan (như MCM/Mộc Châu) ra khỏi phạm vi phân tích của VNM.VN để tránh làm sai lệch điểm cảm xúc.
- **Bộ lọc từ khóa động**: Khi quét tin tức, hệ thống chỉ thu thập tin dựa trên từ khóa khớp chính xác với mã cổ phiếu đang dự báo:
  - **VNM.VN**: `["VNM", "Vinamilk"]`
  - **GOOGL**: `["GOOGL", "Google", "Alphabet"]`
  - **META**: `["META", "Facebook"]`
- **Lọc nguồn RSS CafeF**: Các nguồn tin Việt Nam được lọc từ khóa nghiêm ngặt; tin không chứa từ khóa mục tiêu sẽ bị loại bỏ hoàn toàn.
- **Làm sạch Cache**: Cài đặt lại định dạng năm 4 chữ số đồng bộ (`%Y-%m-%d`) và xóa bỏ các cache lỗi cũ để tái tạo dữ liệu cảm xúc sạch.

### 💾 Tích hợp trực tiếp Cơ sở dữ liệu NewsSentiment:
- Khi có yêu cầu dự báo (`POST /api/predict/trigger/{ticker}`), các tin tức crawl được sẽ được phân tích điểm số cảm xúc (VADER/FinBERT) và tự động ghi vào bảng cơ sở dữ liệu `news_sentiments`.
- Điều này giúp giao diện Web Dashboard hiển thị trực tiếp danh sách bài báo cùng điểm số và nhãn cảm xúc cụ thể tương ứng cho từng mã cổ phiếu.

### ⚙️ Lộ trình Tối ưu hóa AI cốt lõi tiếp theo (Bỏ qua yếu tố Web):
1. **Đồng bộ Kalman Filter**: Đưa Kalman Filter làm mịn giá đóng cửa vào hàm `fetch_and_prepare_data` trong `src/data_loader.py` để đồng nhất dữ liệu chỉ báo kỹ thuật lúc Train và lúc Inference.
2. **Tuning Optuna Độc lập**: Chỉnh sửa `scripts/run_tuning.py` để tìm kiếm siêu tham số tối ưu riêng biệt cho từng mã cổ phiếu (VNM.VN, GOOGL, META) thay vì dùng chung cấu hình của META.
3. **Xây dựng hệ thống Backtest lịch sử**: Viết script giả lập các chiến lược giao dịch thực tế trên tập Test để đo lường lợi nhuận thực tế (Total Return, Sharpe Ratio, Maximum Drawdown).


