# 📑 HƯỚNG DẪN & GHI CHÚ HỆ THỐNG DỰ BÁO GIÁ MỞ CỬA

Tài liệu này ghi nhận toàn bộ cấu trúc dữ liệu, phương pháp huấn luyện, và chi tiết các mô hình AI đang được sử dụng trong dự án nhằm giúp bạn dễ dàng theo dõi và báo cáo.

---

## 1. DỮ LIỆU ĐẦU VÀO (DATA PIPELINE)

### 📊 Nguồn dữ liệu & Đồng nhất đơn vị
Hệ thống sử dụng cơ chế gộp dữ liệu thông minh từ 2 nguồn chính cho mỗi mã cổ phiếu:
1.  **Dữ liệu Vinamilk (VNM.VN):**
    *   **Dữ liệu trường cấp (`data/VNM_prices.csv`):** Từ 17/09/2019 đến 16/03/2026. Đơn vị gốc: Nghìn VNĐ. Hệ thống tự động nhân với 1000 để quy đổi về VNĐ.
    *   **Dữ liệu lịch sử bổ sung (DNSE API):** Tải tự động từ năm 2012 giúp mở rộng tập dữ liệu lên **3.532 phiên giao dịch** (tăng thêm 1.868 phiên).
2.  **Dữ liệu các mã công nghệ Mỹ (GOOGL & META):**
    *   Tải trực tiếp từ **Yahoo Finance API** bắt đầu từ mốc **2010-01-01** (GOOGL) và **2012-05-18** (ngày META IPO).
    *   Đồng nhất đơn vị tiền tệ: Quy đổi toàn bộ giá trị USD sang **VNĐ** theo tỷ giá cố định `1 USD = 25.400 VNĐ`.
    *   Tổng hợp dữ liệu: **4.093 phiên** (GOOGL) và **3.494 phiên** (META).

---

### 📈 Các đặc trưng đầu vào (14 Features)
Để dự báo xu hướng, hệ thống tính toán 14 chỉ báo kỹ thuật quan trọng cho mỗi phiên giao dịch:
*   `close`: Giá đóng cửa của ngày hiện tại.
*   `close_lag1`: Giá đóng cửa của ngày hôm trước (độ trễ 1 phiên).
*   `volume_change`: Tỷ lệ thay đổi khối lượng giao dịch so với phiên trước.
*   `intraday_return`: Tỷ suất sinh lời trong ngày `(Close - Open) / Open`.
*   `volatility_20`: Độ lệch chuẩn của tỷ suất sinh lời trong 20 phiên gần nhất (đo lường biến động).
*   `rsi_14`: Chỉ số sức mạnh tương đối (Relative Strength Index).
*   `MACD_12_26_9`: Chỉ báo trung bình động hội tụ phân kỳ (MACD line).
*   `bb_lower`, `bb_middle`, `bb_upper`: Dải dưới, dải giữa, dải trên của **Bollinger Bands** (chu kỳ 20 phiên, độ lệch 2).
*   `atr_14`: Chỉ báo biên độ dao động thực tế trung bình (**Average True Range**) đo lường độ mạnh của biến động giá.
*   `ema_14`: Đường trung bình di động lũy thừa (EMA 14 phiên) giúp nắm bắt xu hướng giá trơn tru.
*   `roc_10`: Tốc độ thay đổi giá (Rate of Change 10 phiên) đo lường động lượng giá.
*   `adx_14`: Chỉ số định hướng trung bình (ADX 14 phiên) xác định cường độ mạnh/yếu của xu hướng hiện tại.

---

### 🎯 Biến mục tiêu dự báo (Target)
Mô hình dự báo **Tỷ suất lợi nhuận mở cửa ngày mai (`target_return`)**:
$$\text{target\_return} = \frac{\text{Open}_{tomorrow} - \text{Close}_{today}}{\text{Close}_{today}}$$

Sau khi AI dự báo ra tỷ suất này, hệ thống sẽ tự động quy đổi ngược về giá tiền thực tế:
$$\text{Giá mở cửa dự báo} = \text{Close}_{today} \times (1 + \text{target\_return}_{predicted})$$

---

## 2. PHƯƠNG PHÁP HUẤN LUYỆN (TRAINING FLOW)

*   **Tính tái lập (Reproducibility):** Cố định seed toàn cục `SEED = 42` cho Numpy, Random, TensorFlow.
*   **Cửa sổ trượt (Sliding Window):** Sử dụng `time_steps = 45` phiên giao dịch liên tiếp trong quá khứ (~2.5 tháng) làm chuỗi dữ liệu đầu vào.
*   **Chuẩn hóa dữ liệu (Scaling):** Sử dụng `MinMaxScaler` đưa dữ liệu về khoảng `[0, 1]`. Bộ chuẩn hóa đầu vào (`feature_scaler`) và đầu ra (`target_scaler`) được tách riêng biệt.
*   **Chia tập dữ liệu (Split Strategy):** Chia tỷ lệ **80% huấn luyện (Train) / 20% kiểm thử (Test)** theo trật tự thời gian (Chronological Split) để tránh rò rỉ dữ liệu tương lai.

---

## 3. CHI TIẾT CÁC MÔ HÌNH AI

Hệ thống huấn luyện song song hai mô hình tối ưu nhất:

### 1. 🌳 XGBoost (Extreme Gradient Boosting)
*   **Đặc điểm:** Mô hình dạng cây quyết định nâng cao, rất mạnh với dữ liệu dạng bảng. Cần làm phẳng dữ liệu chuỗi 3D thành 2D trước khi đưa vào huấn luyện.
*   **Tối ưu tham số (GridSearchCV):** Áp dụng quét lưới kết hợp kiểm thử chéo chuỗi thời gian (`TimeSeriesSplit` với 5 splits) chạy song song (`n_jobs=-1`) để tự động tìm kiếm bộ siêu tham số tốt nhất.
*   **Bộ tham số tối ưu tìm được:**
    *   **VNM.VN:** `{'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}`
    *   **GOOGL:** `{'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}`
    *   **META:** `{'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}`
*   **Ý nghĩa chi tiết của các siêu tham số:**

| Siêu tham số | Ý nghĩa kỹ thuật | Ý nghĩa thực tế trong dự án |
| :--- | :--- | :--- |
| **`n_estimators`** (100) | Số lượng cây quyết định được tạo lập tuần tự để bổ trợ sai số cho nhau. | 100 cây là con số lý tưởng, giúp mô hình đạt độ hội tụ sai số tối thiểu mà không tiêu tốn tài nguyên tính toán. |
| **`learning_rate`** (0.05) | Tốc độ học (hệ số co hẹp đóng góp của mỗi cây quyết định mới). | Giá trị nhỏ `0.05` giúp mô hình học từ từ qua từng cây, tránh hiện tượng nhảy quá đà (Overfitting) và khớp mịn hơn với xu hướng dài hạn. |
| **`max_depth`** (3 hoặc 5) | Độ sâu tối đa (số tầng phân nhánh tối đa) của mỗi cây quyết định. | Cây nông ở mức 3 (cho GOOGL/META) hoặc 5 (cho VNM.VN) giúp giới hạn độ phức tạp của mỗi cây, ngăn chặn việc cây học thuộc lòng các nhiễu nhỏ trong giá sàn. |
| **`subsample`** (0.8) | Tỷ lệ số dòng dữ liệu (phiên giao dịch) được lấy mẫu ngẫu nhiên để huấn luyện mỗi cây. | Mỗi cây chỉ học trên 80% số phiên ngẫu nhiên. Việc này tạo ra sự đa dạng và giúp mô hình chống chịu tốt hơn trước các đột biến giá ngắn hạn (nhiễu thị trường). |
| **`colsample_bytree`** (0.8 hoặc 1.0) | Tỷ lệ số cột dữ liệu (đặc trưng kỹ thuật) được chọn ngẫu nhiên khi xây dựng mỗi cây. | Rút ngẫu nhiên 80% (VNM/GOOGL) hoặc dùng cả 100% (META) số cột đặc trưng đầu vào giúp mô hình không bị lệ thuộc phiến diện vào một vài chỉ báo kỹ thuật cụ thể. |

### 2. 🤖 Transformer Encoder
*   **Đặc điểm:** Kiến trúc Deep Learning tiên tiến nhất sử dụng cơ chế chú ý (Multi-Head Self-Attention) trực tiếp bắt trọn mối quan hệ phi tuyến phức tạp trong chuỗi thời gian 45 ngày.
*   **Cấu trúc chi tiết:**
    *   Lớp nhúng vị trí thời gian `PositionalEmbedding`.
    *   2 Khối Attention liên tiếp: Mỗi khối gồm **MultiHeadAttention (8 heads, key_dim=128)**, **Dropout (0.2)**, Residual Connections và **Layer Normalization**.
    *   2 Lớp Feed-Forward Networks với 256 nơ-ron kích hoạt `ReLU`.
    *   Lớp Flatten và các lớp Dense trung gian (128, 64, 32 nơ-ron) kết hợp Dropout giảm Overfitting.
    *   Biên dịch với thuật toán tối ưu `Adam` với hệ số học nhỏ `1e-4` và hàm mất mát MSE.
    *   Tích hợp bộ kiểm soát `EarlyStopping` (patience=10) và tự động giảm tốc độ học `ReduceLROnPlateau` để tránh overfitting.

---

## 4. KẾT QUẢ ĐÁNH GIÁ THỰC TẾ (HUẤN LUYỆN ĐỘC LẬP - 2010/2012 TO 2026)

Dưới đây là kết quả sai số MAE và sai số phần trăm (MAPE) trên tập kiểm thử (Test Set) sau khi mở rộng dữ liệu lịch sử đầy đủ và tối ưu hóa sâu:

### 🇻🇳 Vinamilk (VNM.VN) - Đơn vị: VNĐ
*   🌳 **XGBoost:**
    *   *Sai số RMSE:* **396,03 VNĐ**
    *   *Sai số MAE:* **223,30 VNĐ** (Lệch trung bình: **0.37%**) 🟢
*   🤖 **Transformer:**
    *   *Sai số RMSE:* **518,62 VNĐ**
    *   *Sai số MAE:* **401,44 VNĐ** (Lệch trung bình: **0.67%**) 🟢

### 🇺🇸 Alphabet / Google (GOOGL) - Đơn vị: VNĐ & USD
*   🌳 **XGBoost:**
    *   *Sai số RMSE:* **71.989,21 VNĐ** (~$2.83 USD)
    *   *Sai số MAE:* **41.090,33 VNĐ** (~$1.62 USD - Lệch trung bình: **0.85%**) 🟢
*   🤖 **Transformer:**
    *   *Sai số RMSE:* **104.386,36 VNĐ** (~$4.11 USD)
    *   *Sai số MAE:* **74.021,98 VNĐ** (~$2.91 USD - Lệch trung bình: **1.52%**) 🟢

### 🇺🇸 Meta Platforms (META) - Đơn vị: VNĐ & USD
*   🌳 **XGBoost:**
    *   *Sai số RMSE:* **859.668,74 VNĐ** (~$33.85 USD)
    *   *Sai số MAE:* **747.162,18 VNĐ** (~$29.42 USD - Lệch trung bình: **4.89%**)
*   🤖 **Transformer:**
    *   *Sai số RMSE:* **853.236,85 VNĐ** (~$33.59 USD)
    *   *Sai số MAE:* **741.061,81 VNĐ** (~$29.18 USD - Lệch trung bình: **4.95%**)

---

## 🛠️ HƯỚNG DẪN CHẠY PIPELINE TRÊN TERMINAL

Mỗi khi muốn chạy lại toàn bộ quá trình tải dữ liệu, tính toán đặc trưng, huấn luyện các mô hình và xuất dự báo cho ngày mai, bạn chỉ cần gõ lệnh:

```powershell
python main.py
```

*   **Kết quả đầu ra:** Các biểu đồ so sánh dự báo của các mô hình được lưu độc lập cho từng mã: `results/model_battle_result_VNM.VN.png`, `results/model_battle_result_GOOGL.png`, và `results/model_battle_result_META.png`.
*   **Các mô hình đã huấn luyện** được tự động lưu trong thư mục `models/` để sử dụng dự báo nhanh mà không cần huấn luyện lại.
*   **Để dọn dẹp các file rác phát sinh** (như Python cache, Jupyter checkpoints), bạn có thể chạy:
```powershell
python clean_workspace.py
```
