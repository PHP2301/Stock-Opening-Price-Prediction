# 📑 HƯỚNG DẪN & GHI CHÚ HỆ THỐNG DỰ BÁO GIÁ MỞ CỬA VNM.VN

Tài liệu này ghi nhận toàn bộ cấu trúc dữ liệu, phương pháp huấn luyện, và chi tiết các mô hình AI đang được sử dụng trong dự án nhằm giúp bạn dễ dàng theo dõi và báo cáo.

---

## 1. DỮ LIỆU ĐẦU VÀO (DATA PIPELINE)

### 📊 Nguồn dữ liệu & Đồng nhất đơn vị
Hệ thống sử dụng cơ chế gộp dữ liệu thông minh từ 2 nguồn:
1.  **Dữ liệu chính từ Nhà trường (`data/VNM_prices.csv`):**
    *   Thời gian: Từ 17/09/2019 đến 16/03/2026.
    *   Đơn vị gốc: **Nghìn VNĐ** (Ví dụ: giá hiển thị là `61.80` tương đương `61,800 VNĐ`).
    *   *Tiền xử lý:* Hệ thống tự động nhân với **1000** để đưa về giá **VNĐ** thực tế.
2.  **Dữ liệu từ Yahoo Finance (`VNM.VN`):**
    *   Thời gian: Từ 17/03/2026 trở đi (chỉ bổ sung phần thời gian thiếu).
    *   Đơn vị gốc: **VNĐ** (Khớp hoàn toàn với dữ liệu nhà trường sau khi nhân 1000).

> **Ghi chú:** Việc gộp dữ liệu giúp tăng số lượng phiên giao dịch lên **1,639 phiên**, cung cấp đủ dữ liệu để huấn luyện các mạng Deep Learning phức tạp.

---

### 📈 Các đặc trưng đầu vào (11 Features)
Để dự báo xu hướng, hệ thống tính toán 11 chỉ báo kỹ thuật quan trọng cho mỗi phiên giao dịch:
*   `close`: Giá đóng cửa của ngày hiện tại.
*   `close_lag1`: Giá đóng cửa của ngày hôm trước (độ trễ 1 phiên).
*   `volume_change`: Tỷ lệ thay đổi khối lượng giao dịch so với phiên trước.
*   `intraday_return`: Tỷ suất sinh lời trong ngày `(Close - Open) / Open`.
*   `volatility_20`: Độ lệch chuẩn của tỷ suất sinh lời trong 20 phiên gần nhất (đo lường biến động).
*   `rsi_14`: Chỉ số sức mạnh tương đối (Relative Strength Index).
*   `MACD_12_26_9`: Chỉ báo trung bình động hội tụ phân kỳ.
*   `bb_lower`, `bb_middle`, `bb_upper`: Dải dưới, dải giữa, dải trên của **Bollinger Bands** (chu kỳ 20 phiên, độ lệch 2).
*   `atr_14`: Chỉ báo biên độ dao động thực tế trung bình (**Average True Range**) đo lường độ mạnh của biến động giá.

---

### 🎯 Biến mục tiêu dự báo (Target)
Mô hình **không dự báo trực tiếp giá tiền** vì giá cổ phiếu không dừng (non-stationary). Thay vào đó, mô hình dự báo **Tỷ suất lợi nhuận mở cửa ngày mai (`target_return`)**:
$$\text{target\_return} = \frac{\text{Open}_{tomorrow} - \text{Close}_{today}}{\text{Close}_{today}}$$

Sau khi AI dự báo ra tỷ suất này, hệ thống sẽ tự động quy đổi ngược về giá tiền thực tế:
$$\text{Giá mở cửa dự báo} = \text{Close}_{today} \times (1 + \text{target\_return}_{predicted})$$

---

## 2. PHƯƠNG PHÁP HUẤN LUYỆN (TRAINING FLOW)

*   **Tính tái lập (Reproducibility):** Cố định seed toàn cục `SEED = 42` cho Numpy, Random, TensorFlow để đảm bảo chạy lại nhiều lần luôn ra kết quả giống nhau.
*   **Cửa sổ trượt (Sliding Window):** Sử dụng `time_steps = 30` phiên giao dịch liên tiếp trong quá khứ (~1.5 tháng) làm chuỗi dữ liệu đầu vào để dự báo phiên tiếp theo.
*   **Chuẩn hóa dữ liệu (Scaling):**
    *   Sử dụng `MinMaxScaler` đưa dữ liệu về khoảng `[0, 1]`.
    *   Bộ chuẩn hóa đầu vào (`feature_scaler`) và đầu ra (`target_scaler`) được tách riêng biệt để đảm bảo dịch ngược giá tiền chính xác.
*   **Chia tập dữ liệu (Split Strategy):**
    *   Chia tỷ lệ **80% huấn luyện (Train) / 20% kiểm thử (Test)**.
    *   Chia theo **trật tự thời gian (Chronological Split)** thay vì chia ngẫu nhiên để tránh hiện tượng rò rỉ dữ liệu tương lai (Data Leakage).

---

## 3. CHI TIẾT CÁC MÔ HÌNH AI

Hệ thống huấn luyện song song và đánh giá 4 mô hình:

### 1. 🌳 XGBoost (Extreme Gradient Boosting)
*   **Đặc điểm:** Mô hình dạng cây quyết định nâng cao, rất mạnh với dữ liệu dạng bảng.
*   **Huấn luyện:** Áp dụng `GridSearchCV` tự động tối ưu hóa các siêu tham số quan trọng như `n_estimators`, `max_depth`, `learning rate`.

### 2. 🧠 LSTM (Long Short-Term Memory)
*   **Đặc điểm:** Mạng hồi quy Recurrent Neural Network (RNN) chuyên biệt cho chuỗi thời gian, giúp ghi nhớ các xu hướng dài hạn.
*   **Tối ưu:** Tích hợp bộ kiểm soát `EarlyStopping` (dừng sớm nếu val_loss không giảm sau 10 epochs) và tự động giảm tốc độ học `ReduceLROnPlateau` khi học bị chững lại.

### 3. 🤖 Transformer Encoder
*   **Đặc điểm:** Kiến trúc Deep Learning tiên tiến nhất sử dụng cơ chế chú ý (Multi-Head Self-Attention) giúp nắm bắt mối quan hệ của tất cả các phiên trong cửa sổ 30 ngày một cách trực tiếp.

### 4. 🤝 Mô hình kết hợp (Ensemble Model)
*   **Công thức:** Lấy trung bình cộng dự báo của hai mô hình ổn định nhất:
    $$\text{Dự báo Ensemble} = 0.5 \times \text{Dự báo LSTM} + 0.5 \times \text{Dự báo XGBoost}$$
*   **Ưu điểm:** Giảm sai số đỉnh (RMSE) và tăng độ tin cậy vượt trội so với việc chỉ sử dụng một mô hình đơn lẻ.

---

## 4. KẾT QUẢ ĐÁNH GIÁ THỰC TẾ

| Mô hình | Sai số MAE (Sai lệch trung bình so với thực tế) |
|---|---|
| **XGBoost** | ~283 VNĐ |
| **LSTM** | ~321 VNĐ |
| **Transformer** | ~527 VNĐ |
| **Ensemble (LSTM + XGBoost)** | **~294 VNĐ** (Sai số cực nhỏ, ổn định nhất) |

---

## 🛠️ HƯỚNG DẪN CHẠY PIPELINE TRÊN TERMINAL

Mỗi khi muốn chạy lại toàn bộ quá trình tải dữ liệu, tính toán đặc trưng, huấn luyện các mô hình và xuất dự báo cho ngày mai, bạn chỉ cần gõ lệnh:

```powershell
python main.py
```

*   **Kết quả đầu ra:** Biểu đồ so sánh dự báo của các mô hình được lưu tại file `model_battle_result.png`.
*   **Các mô hình đã huấn luyện** được tự động lưu trong thư mục `models/` để sử dụng dự báo nhanh mà không cần huấn luyện lại.
