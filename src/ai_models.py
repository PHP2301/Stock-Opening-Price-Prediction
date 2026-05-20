import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ẩn các cảnh báo hệ thống của TensorFlow
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from xgboost import XGBRegressor

# ==========================================
# 1. MÔ HÌNH XGBOOST (Cần làm phẳng dữ liệu 3D thành 2D)
# ==========================================
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def build_xgboost_optimized(X_train_flat, y_train):
    """
    Sử dụng thuật toán GridSearchCV kết hợp TimeSeriesSplit 
    để tự động tìm bộ tham số tối ưu nhất cho XGBoost.
    """
    xgb_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    # Định nghĩa "lưới" các tham số cần thử nghiệm
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    # Chia tập kiểm thử chéo dạng chuỗi thời gian (không làm xáo trộn ngày tháng)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Khởi động thuật toán quét ma trận tham số
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1
    )
    
    print("⏳ Đang quét tìm bộ tham số tốt nhất cho XGBoost...")
    grid_search.fit(X_train_flat, y_train.ravel())
    
    print(f"🔥 Tham số tối ưu tìm được: {grid_search.best_params_}")
    return grid_search.best_estimator_

# ==========================================
# 2. MÔ HÌNH LSTM (Long Short-Term Memory)
# ==========================================
def build_lstm(input_shape):
    """
    Xây dựng mạng nơ-ron LSTM chuyên sâu cho chuỗi thời gian tài chính.
    input_shape: (time_steps, features) -> Hiện tại là (30, 5)
    """
    model = Sequential([
        Input(shape=input_shape),
        # Lớp LSTM đầu tiên, trả về toàn bộ chuỗi để lớp sau học tiếp
        LSTM(units=64, return_sequences=True),
        Dropout(0.2), # Ngắt ngẫu nhiên 20% nơ-ron chống Overfitting
        
        # Lớp LSTM thứ hai, nén thông tin chuỗi lại
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        
        # Lớp liên kết hoàn toàn (Dense) để đưa ra 1 con số giá mở cửa duy nhất
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])
    
    # Biên dịch mô hình với thuật toán tối ưu Adam và hàm mất mát MSE
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ==========================================
# 3. MÔ HÌNH TRANSFORMER (Encoder Only)
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Một khối Encoder tiêu chuẩn của Transformer sử dụng cơ chế Self-Attention"""
    # Lớp chú ý đa đầu (Multi-Head Self-Attention)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    res = LayerNormalization(epsilon=1e-6)(x + inputs) # Residual Connection + Chuẩn hóa lớp

    # Lớp Feed Forward (Mạng truyền thẳng)
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return LayerNormalization(epsilon=1e-6)(x + res)
class PositionalEmbedding(tf.keras.layers.Layer):
    """Lớp toán học gán thẻ thứ tự thời gian cho dữ liệu chuỗi"""
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=d_model)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.pos_emb(positions)
        return inputs + embedded_positions

def build_transformer(input_shape):
    """Xây dựng kiến trúc mạng Transformer hoàn chỉnh để dự báo chuỗi số"""
    inputs = Input(shape=input_shape)
    
    # Chạy qua một khối Transformer Encoder
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=64, dropout=0.2)
    
    # Nén không gian thời gian (30 phiên) về dạng phẳng cố định
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    
    # Lớp Dense đầu ra để dự báo giá
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Chạy kiểm thử độc lập xem TensorFlow có dựng được khung xương mô hình mà không lỗi cấu trúc
    print("=== KIỂM THỬ KIẾN TRÚC MÔ HÌNH AI ===")
    sample_shape = (30, 7)
    
    lstm_m = build_lstm(sample_shape)
    print(f"✅ Khởi tạo mạng LSTM thành công! Tổng số lớp: {len(lstm_m.layers)}")
    
    trans_m = build_transformer(sample_shape)
    print(f"✅ Khởi tạo mạng Transformer thành công! Đầu ra mô hình: {trans_m.output_shape}")