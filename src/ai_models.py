import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ẩn các cảnh báo hệ thống của TensorFlow
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Flatten
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
    xgb_model = XGBRegressor(random_state=42, n_jobs=1)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
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
# 3. MÔ HÌNH TRANSFORMER (Encoder Only)
# ==========================================
@tf.keras.utils.register_keras_serializable()
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config

def build_transformer(input_shape):
    """
    Xây dựng kiến trúc mạng Transformer sâu hơn để học tốt hơn trên chuỗi thời gian:
    - 2 Lớp Multi-Head Attention (8 heads, key_dim=128)
    - Residual Connections & Layer Normalization
    - Feed-Forward Networks cho từng block
    - Tăng số neuron lớp Dense để khớp dữ liệu phức tạp
    """
    inputs = Input(shape=input_shape)
    
    # 1. Project input features to hidden size d_model
    d_model = 128
    x = Dense(d_model)(inputs)
    
    # 2. Add Positional Embedding
    x = PositionalEmbedding(sequence_length=input_shape[0], d_model=d_model)(x)
    
    # --- BLOCK 1 ---
    # Multi-Head Attention
    attn_1 = MultiHeadAttention(key_dim=128, num_heads=8, dropout=0.2)(x, x)
    attn_1 = Dropout(0.2)(attn_1)
    x = LayerNormalization(epsilon=1e-6)(x + attn_1)  # Residual connection 1
    
    # Feed Forward Network 1
    ffn_1 = Dense(256, activation="relu")(x)
    ffn_1 = Dropout(0.2)(ffn_1)
    ffn_1 = Dense(d_model)(ffn_1)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_1)  # Residual connection 2
    
    # --- BLOCK 2 ---
    # Multi-Head Attention
    attn_2 = MultiHeadAttention(key_dim=128, num_heads=8, dropout=0.2)(x, x)
    attn_2 = Dropout(0.2)(attn_2)
    x = LayerNormalization(epsilon=1e-6)(x + attn_2)  # Residual connection 3
    
    # Feed Forward Network 2
    ffn_2 = Dense(256, activation="relu")(x)
    ffn_2 = Dropout(0.2)(ffn_2)
    ffn_2 = Dense(d_model)(ffn_2)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_2)  # Residual connection 4
    
    # 4. Flatten to retain complete temporal structure
    x = Flatten()(x)
    
    # 5. Output layers
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    
    # Use a smaller learning rate to prevent early overfitting on scaled return targets
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

if __name__ == "__main__":
    import sys
    import os
    # Cấu hình UTF-8 cho console để in tiếng Việt không lỗi trên Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Run independent architecture compilation tests
    print("=== TESTING AI MODEL ARCHITECTURES ===")
    sample_shape = (45, 14)
    
    
    trans_m = build_transformer(sample_shape)
    print(f"Transformer model built successfully! Output shape: {trans_m.output_shape}")