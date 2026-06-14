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

### 📈 Các đặc trưng đầu vào (42 Features)

Để dự báo xu hướng, hệ thống tính toán 42 chỉ báo kỹ thuật, vĩ mô, cổ tức và dòng tiền dạng dừng (stationary features) sau khi làm mịn qua bộ lọc nhiễu **Kalman Filter**:

1.  **Nhánh 1: Giá & Động lượng (12 đặc trưng):**
    - `gap_open`: Chênh lệch giá mở cửa hôm nay so với đóng cửa hôm trước.
    - `open_return`: Tỷ suất sinh lời mở cửa.
    - `buying_pressure`: Áp lực mua trong phiên.
    - `shadow_ratio`: Tỷ lệ bóng nến trên/dưới.
    - `intraday_range`: Biên độ dao động giá nội phiên.
    - `return_1d`, `return_2d`, `return_3d`: Tỷ suất sinh lời đóng cửa các phiên trước.
    - `mom_5d`, `mom_10d`, `mom_20d`: Chỉ báo động lượng động thái giá.
    - `dist_ma50`: Khoảng cách tương đối từ giá hiện tại đến đường SMA 50.
2.  **Nhánh 2: Khối lượng & Biến động (6 đặc trưng):**
    - `volume_change`: Tốc độ thay đổi khối lượng giao dịch.
    - `volume_sma_ratio`: Khối lượng hiện tại so với trung bình 20 phiên.
    - `volume_zscore`: Điểm chuẩn hóa Z-score khối lượng (kích hoạt slippage động).
    - `ad_line_ratio`: Chỉ báo tích lũy/phân phối chuẩn hóa.
    - `obv_zscore`: Z-score của chỉ số khối lượng cân bằng tích lũy OBV.
    - `vol_ratio`: Biến động khối lượng tương đối.
3.  **Nhánh 3: Kỹ thuật, Vĩ mô & Lịch (16 đặc trưng):**
    - `rsi_14`: Chỉ số sức mạnh tương đối.
    - `macd_ratio`: Tỷ lệ đường MACD trên đường tín hiệu Signal.
    - `bb_position`: Vị trí tương đối của giá trong dải Bollinger Bands.
    - `adx_14`: Chỉ số định hướng trung bình (đo sức mạnh xu hướng).
    - `stoch_k`: Chỉ báo dao động ngẫu nhiên Stochastic %K.
    - `efficiency_ratio`: Chỉ số hiệu quả Kaufman (đo lường độ nhiễu của giá).
    - `vix_lag1`: Chỉ số biến động VIX trễ 1 ngày.
    - `bond_yield_lag1`: Lợi suất trái phiếu chính phủ Mỹ 10 năm trễ 1 ngày.
    - `usdvnd_change`: Biến động tỷ giá USD/VND.
    - `vnindex_return_lag1`: Lợi suất chỉ số VN-Index (đối với mã Việt Nam).
    - `day_of_week_sin` / `cos` / `month_sin` / `cos`: Mã hóa dạng sóng tuần hoàn thời gian.
    - `is_quarter_end`: Đánh dấu các ngày chốt sổ cuối quý.
    - `days_before_tet`: Số ngày đếm ngược đến Tết Nguyên Đán (áp dụng riêng cho VNM.VN).
4.  **Nhánh 4: Dòng tiền & Cổ tức (8 đặc trưng):**
    - `mfi_14`: Chỉ số dòng tiền Money Flow Index.
    - `dividend_flag`: Tín hiệu ngày không hưởng quyền (ex-dividend date).
    - `days_to_dividend`: Số ngày đếm ngược đến kỳ cổ tức ti�## 2. PHƯƠNG PHÁP HUẤN LUYỆN (TRAINING FLOW)

- **Tính tái lập (Reproducibility):** Cố định seed toàn cục `SEED = 42` cho Numpy, Random, TensorFlow.
- **Cửa sổ trượt (Sliding Window):** Sử dụng `time_steps = 45` phiên giao dịch liên tiếp trong quá khứ (~2.5 tháng) làm chuỗi dữ liệu đầu vào.
- **Chuẩn hóa dữ liệu (Scaling):** Sử dụng `StandardScaler` (cho cả đặc trưng đầu vào `feature_scaler` và biến mục tiêu `target_scaler`) giúp chuyển đổi dữ liệu về dạng có trung bình = 0 và độ lệch chuẩn = 1.
- **Chia tập dữ liệu (Split Strategy):** Chia tỷ lệ **80% huấn luyện (Train) / 20% kiểm thử (Test)** theo trật tự thời gian (Chronological Split) với Purge Gap = 45 phiên để tránh rò rỉ dữ liệu tương lai.

---

## 3. CHI TIẾT CÁC MÔ HÌNH AI

Hệ thống huấn luyện song song hai mô hình tối ưu nhất:

### 1. 🌳 XGBoost (Extreme Gradient Boosting)

- **Đặc điểm:** Mô hình dạng cây quyết định nâng cao, rất mạnh với dữ liệu dạng bảng. XGBoost trong mô hình lai (Hybrid model) nhận đầu vào là sự kết hợp của **42 đặc trưng thô + 40 đặc trưng ẩn (latent features)** được trích xuất từ Transformer, tạo thành vector đầu vào **82 chiều**.
- **Tối ưu tham số (GridSearchCV):** Áp dụng quét lưới kết hợp kiểm thử chéo chuỗi thời gian (`TimeSeriesSplit` với 5 splits) để tự động tìm kiếm bộ siêu tham số tốt nhất.
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

### 2. 🤖 Branched Multi-Task Transformer

- **Đặc điểm:** Kiến trúc Deep Learning đa nhiệm tiên tiến nhất sử dụng cơ chế chú ý (Multi-Head Self-Attention) phân nhánh độc lập tương ứng với 4 nhóm đặc trưng đầu vào để trích xuất 40 chiều đặc trưng ẩn (latent embedding).
- **Cấu trúc chi tiết:**
  - **Nhánh đầu vào phân nhánh (4 Branches):** Mỗi nhánh (Giá, Khối lượng, Kỹ thuật, Dòng tiền/Cổ tức) đi qua lớp Conv1D (64 filters) và Dense layer độc lập để tạo ra không gian biểu diễn riêng.
  - **Lớp Attention chung (Multi-Head Self-Attention):** Sử dụng 8 heads (`num_heads=8`, `key_dim=64`) để tính toán mối quan hệ tuần hoàn và liên kết dài hạn giữa các ngày khác nhau trong cửa sổ trượt 45 phiên.
  - **Hai đầu ra đa nhiệm (Multi-task Heads):**
    - `return_head`: Dự báo tỷ suất lợi nhuận mở cửa (T+1, T+2, T+3).
    - `spread_head`: Dự báo độ rộng chênh lệch mở/đóng (T+1, T+2, T+3).
  - **Bộ cân bằng trọng số tự động (Uncertainty Weighting):** Sử dụng lớp tùy chỉnh `UncertaintyWeightsLayer` để tự động điều chỉnh trọng số hàm mất mát (Huber Loss) của 2 đầu ra trong quá trình lan truyền ngược gradient.
  - **Biên dịch và điều phối học:** Sử dụng thuật toán tối ưu `Adam` (tốc độ học từ `1e-4` giảm dần qua Cosine Decay), tích hợp `EarlyStopping` và `ReduceLROnPlateau`.

---            |
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

### ⚙️ Cải Tiến & Đồng Bộ Hệ Thống (Cập nhật 13/06/2026)

Hệ thống đã hoàn thành việc đồng bộ hóa và nâng cấp AI cốt lõi bao gồm:
1. **Đồng bộ Kalman Filter**: Tích hợp bộ lọc Kalman làm mịn giá trực tiếp vào hàm `fetch_and_prepare_data` trong `src/data_loader.py`.
2. **Tuning Optuna Độc lập**: Tối ưu hóa siêu tham số riêng biệt cho từng mã cổ phiếu, cải thiện đáng kể độ khớp.
3. **Dự Báo & Mô Hình Đa Mục Tiêu (Multi-Task Learning)**: Chuyển đổi mô hình sang dự báo chuỗi 3 ngày liên tiếp (T+1, T+2, T+3) cho cả biến Tỷ suất sinh lời (Return) và Độ rộng chênh lệch (Spread), sử dụng cơ chế tự động cân bằng trọng số hàm mất mát (Kendall uncertainty weighting).
4. **Hệ Thống Kiểm Định Cuốn Chiếu Rolling Walk-Forward Backtesting**: Triển khai kịch bản giao dịch thực tế mô phỏng năm-theo-năm (2023–2026) với cơ chế tự động tái huấn luyện cuốn chiếu (`scripts/run_walk_forward_backtest.py`), ngăn chặn hiện tượng rò rỉ dữ liệu (scaling isolation) và tối ưu hóa thời gian huấn luyện qua cơ chế warm-start weights.

### 📊 Kết quả Backtest Walk-Forward Cuốn Chiếu (2023 - 2026)
*Tham số giao dịch qua đêm (Overnight): Phí VN = 0.20%, Phí US = 0.15%, Trượt giá = 0.10%, Ngưỡng tín hiệu = 0.10%*

- **GOOGL (Alphabet)**: Đạt hiệu quả vượt trội. Chiến lược đạt lợi nhuận **+100.75%** (Sharpe **1.39**, Max Drawdown **-16.68%** trên 53 lệnh giao dịch). Việc tự động cập nhật trọng số hàng năm giúp mô hình xóa bỏ hoàn toàn bias âm từ dữ liệu lịch sử và tận dụng đà tăng trưởng mạnh mẽ của kỷ nguyên AI.
- **META**: Đạt lợi nhuận chiến lược **+23.66%** (Sharpe **0.38**, Max Drawdown **-37.97%** trên 56 lệnh). Mô hình hoạt động cực tốt trong chu kỳ phục hồi năm 2023.
- **VNM.VN**: Ghi nhận lợi nhuận chiến lược **-37.06%** (Sharpe **-1.25** trên 39 lệnh). Điều này củng cố phân tích trước đó rằng VNM.VN có tín hiệu tương quan rất yếu (+0.03), việc giao dịch liên tục chỉ làm hao mòn tài sản qua phí và trượt giá, xác nhận chiến lược đối với VNM.VN không có Edge thực tế.

Để chạy giả lập kiểm định cuốn chiếu cho một mã cổ phiếu:
```powershell
python scripts/run_walk_forward_backtest.py GOOGL
```
Đồ thị so sánh đường cong tài sản tích lũy (Equity Curve) sẽ tự động được xuất ra tại thư mục `reports/figures/walk_forward_equity_curve_{ticker}.png`.


