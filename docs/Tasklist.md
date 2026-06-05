# BẢNG TIẾN ĐỘ XÂY DỰNG MÔ HÌNH LAI (HYBRID MODEL)

File file này giúp bạn theo dõi các đầu việc cần làm để xây dựng và thử nghiệm kiến trúc lai **Transformer/LSTM + XGBoost**. Bạn có thể đánh dấu `[x]` vào các ô tương ứng khi hoàn thành xong.

> [!IMPORTANT]
> **Trọng tâm hiện tại (Active Focus):** Tập trung tối ưu hóa và huấn luyện riêng biệt mô hình **Transformer** trước. Chỉ khi mô hình Transformer đạt độ chính xác cao nhất và biểu đồ Loss hội tụ hoàn hảo, chúng ta mới bắt đầu trích xuất đặc trưng để huấn luyện XGBoost.
> * Lệnh chạy huấn luyện riêng Transformer: `python scripts/run_training_transformer.py`

---

## 📋 Danh sách công việc (Task Checklist)

### Giai đoạn 1: Chuẩn bị & Chia dữ liệu (Đã có sẵn trong Pipeline)
- [x] **1.1.** Tải dữ liệu đặc trưng đầy đủ (24 đặc trưng) của các mã cổ phiếu (`VNM.VN`, `GOOGL`, `META`). *(Đã có sẵn ở `fetch_and_prepare_data`)*
- [x] **1.2.** Chia dữ liệu thành tập Train (80%) và Test (20%) theo thứ tự thời gian. *(Đã có sẵn ở `DataTransformer.split_train_test_chronological`)*
- [x] **1.3.** Chuẩn hóa dữ liệu bằng `StandardScaler()`. *(Đã có sẵn ở `DataTransformer.fit_transform_data`)*
- [x] **1.4.** Chuyển đổi dữ liệu 2D phẳng thành cấu trúc 3D với lookback window = 45 phiên. *(Đã có sẵn ở `DataTransformer.create_sliding_windows`)*

### Giai đoạn 2: Xây dựng & Huấn luyện Model tuần tự (Đã có sẵn trong `ai_models.py`)
- [x] **2.1.** Thiết kế mô hình mạng nơ-ron nhận đầu vào shape `(45, 24)`. *(Đã có trong `build_transformer`)*
- [x] **2.2.** Khai báo lớp Dense áp chót có kích thước nhỏ (lớp `Dense(32)` trong code của bạn).
- [x] **2.3.** Khai báo lớp Dense cuối cùng `Dense(1)` để dự đoán tỷ suất lợi nhuận. *(Đã có trong `build_transformer`)*
- [x] **2.4.** Huấn luyện mô hình trên tập Train để có file `.keras` lưu ở thư mục `models/` *(Đã hoàn thành, các file `transformer_model_*.keras` đã có sẵn trong thư mục `models/`)*.

### Giai đoạn 3: Trích xuất Đặc trưng & Ghép nối dữ liệu
- [x] **3.1.** Khởi tạo mô hình phụ `feature_extractor` lấy đầu ra từ lớp áp chót. *(Đã thực hiện trong `hybrid_main.py`)*
- [x] **3.2.** Chạy `feature_extractor.predict()` để lấy đặc trưng chuỗi thời gian 2D (`N_samples, 32`). *(Đã hoàn thành)*
- [x] **3.3.** Trích xuất các chỉ báo kỹ thuật của ngày hiện tại (phiên thứ 45). *(Đã hoàn thành)*
- [x] **3.4.** Ghép nối (concatenate) đặc trưng ẩn (32 chiều) với chỉ báo kỹ thuật ngày hiện tại (24 chiều) tạo ra 56 đặc trưng lai. *(Đã hoàn thành)*

### Giai đoạn 4: Huấn luyện XGBoost & Đánh giá (Đặc biệt: Stacking trên 10% holdout)
- [x] **4.1.** Khởi tạo mô hình `XGBRegressor()` với các tham số tối ưu (đã quét GridSearch tự động). *(Đã hoàn thành)*
- [x] **4.2.** Huấn luyện XGBoost trên tập dữ liệu lai phẳng mới. *(Đã huấn luyện trên 10% dữ liệu holdout của Transformer để chống Overfitting chéo)*
- [x] **4.3.** Dự đoán trên tập Test lai và đánh giá các chỉ số sai số (RMSE, MAE, MAPE). *(Đã hoàn thành)*
- [x] **4.4.** Vẽ đồ thị so sánh thực tế vs dự đoán để trực quan hóa kết quả. *(Đã hoàn thành)*


---

## 💡 Gợi ý Code mẫu từng phần (Starter Code)

### 1. Dựng Transformer/LSTM trích xuất đặc trưng (Giai đoạn 2)
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_hybrid_lstm(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layers
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32, return_sequences=False)(x)
    
    # LỚP ÁP CHÓT: Trích xuất đặc trưng (16 chiều)
    latent = layers.Dense(16, activation="relu", name="latent_features")(x)
    
    # Lớp đầu ra để huấn luyện
    outputs = layers.Dense(1, activation="linear", name="prediction_output")(latent)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
```

### 2. Trích xuất đặc trưng & Ghép nối (Giai đoạn 3)
```python
# 1. Trích xuất đặc trưng chuỗi thời gian từ mạng đã học
feature_extractor = Model(inputs=model.input, outputs=model.get_layer("latent_features").output)
X_train_latent = feature_extractor.predict(X_train_3D) # shape: (N_train, 16)

# 2. Lấy các chỉ báo thủ công của ngày hiện tại (cột cuối cùng trong window 45 ngày)
# Giả sử X_train_3D có hình dạng [N, 45, 23], ngày cuối cùng là X_train_3D[:, -1, :]
X_train_indicators = X_train_3D[:, -1, :] # shape: (N_train, 23)

# 3. Ghép nối (Concat) thành tập dữ liệu lai phẳng cho XGBoost
import numpy as np
X_train_hybrid = np.concatenate([X_train_latent, X_train_indicators], axis=1) # shape: (N_train, 39)
```

### 3. Huấn luyện XGBoost (Giai đoạn 4)
```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Khởi tạo và fit mô hình
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train_hybrid, y_train)

# Dự đoán
y_pred = xgb_model.predict(X_test_hybrid)
```

---

## ⚙️ Giai đoạn 5: Tối ưu hóa lõi AI chuyên sâu (Chuẩn bị thực hiện)
- [ ] **5.1.** Đồng bộ bộ lọc Kalman Filter vào tệp chuẩn bị dữ liệu huấn luyện (`src/data_loader.py`).
- [ ] **5.2.** Cấu hình tìm kiếm siêu tham số (Optuna Tuning) độc lập cho mỗi mã cổ phiếu trong `scripts/run_tuning.py` thay vì dùng chung cấu hình của META.
- [ ] **5.3.** Phát triển module Backtesting mô phỏng giao dịch thực tế trên tập Test để đo hiệu suất lợi nhuận đầu tư và các hệ số tài chính (Sharpe, Drawdown).
