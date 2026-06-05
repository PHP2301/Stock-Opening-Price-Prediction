import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ẩn các cảnh báo hệ thống của TensorFlow
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Flatten, Conv1D, Bidirectional, GRU, Multiply
from xgboost import XGBRegressor

# ==========================================
# 1. MÔ HÌNH XGBoost (Cần làm phẳng dữ liệu 3D thành 2D)
# ==========================================
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

def build_xgboost_optimized(X_train_flat, y_train):
    """
    Sử dụng thuật toán RandomizedSearchCV kết hợp TimeSeriesSplit 
    để tự động tìm bộ tham số tối ưu nhất cho XGBoost một cách nhanh chóng.
    """
    # Đặt n_jobs=1 cho XGBRegressor đơn lẻ để tránh tranh chấp tài nguyên với n_jobs=-1 của RandomizedSearchCV
    xgb_model = XGBRegressor(random_state=42, n_jobs=1)
    
    # Định nghĩa không gian tham số rộng hơn
    param_distributions = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Chia tập kiểm thử chéo dạng chuỗi thời gian (không làm xáo trộn ngày tháng)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Sử dụng RandomizedSearchCV để thử ngẫu nhiên 6 cấu hình tối ưu nhất
    search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=param_distributions, 
        n_iter=6,
        cv=tscv, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1,
        random_state=42
    )
    
    print("⏳ Đang quét nhanh bộ tham số tối ưu cho XGBoost...")
    search.fit(X_train_flat, y_train.ravel())
    
    print(f"🔥 Tham số tối ưu tìm được: {search.best_params_}")
    return search.best_estimator_


# ==========================================
# 2. MÔ HÌNH TRANSFORMER (Encoder Only)
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

def glu(x, d_model):
    gate = Dense(d_model, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    linear = Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return Multiply()([linear, gate])

def build_transformer(input_shape, d_model=128, num_heads=8, dropout_rate=0.3, learning_rate=1e-4, multi_task=True):
    """
    Xây dựng kiến trúc mạng Transformer sâu hơn để học tốt hơn trên chuỗi thời gian:
    - 2 Lớp Multi-Head Attention (num_heads, key_dim=d_model, dropout=dropout_rate)
    - Residual Connections & Layer Normalization
    - Gated Linear Unit (GLU) ở cổng vào để lọc đặc trưng
    - Bidirectional GRU thay cho Flatten để tóm tắt thông tin thời gian mà không phá vỡ trình tự
    - Tăng số neuron lớp Dense để khớp dữ liệu phức tạp và chống overfitting
    """
    inputs = Input(shape=input_shape)
    
    # 1. Conv1D layer to extract local patterns
    x = Conv1D(
        filters=d_model, 
        kernel_size=3, 
        padding='same', 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Apply GLU gating to filter inputs
    x = glu(x, d_model)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # 2. Add Positional Embedding
    x = PositionalEmbedding(sequence_length=input_shape[0], d_model=d_model)(x)
    
    # --- BLOCK 1 ---
    # Multi-Head Attention
    attn_1 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads, dropout=dropout_rate)(x, x)
    attn_1 = Dropout(dropout_rate)(attn_1)
    x = LayerNormalization(epsilon=1e-6)(x + attn_1)  # Residual connection 1
    
    # Feed Forward Network 1
    ffn_1 = Dense(2 * d_model, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    ffn_1 = Dropout(dropout_rate)(ffn_1)
    ffn_1 = Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ffn_1)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_1)  # Residual connection 2
    
    # --- BLOCK 2 ---
    # Multi-Head Attention
    attn_2 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads, dropout=dropout_rate)(x, x)
    attn_2 = Dropout(dropout_rate)(attn_2)
    x = LayerNormalization(epsilon=1e-6)(x + attn_2)  # Residual connection 3
    
    # Feed Forward Network 2
    ffn_2 = Dense(2 * d_model, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    ffn_2 = Dropout(dropout_rate)(ffn_2)
    ffn_2 = Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ffn_2)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_2)  # Residual connection 4
    
    # 4. Bidirectional GRU instead of Flatten
    x = Bidirectional(
        GRU(d_model // 4, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    )(x)
    
    # 5. Output layers
    x = Dense(d_model, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(d_model // 2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    
    # Shared latent embedding space
    embedding = Dense(32, activation="relu", name="latent_embedding", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    
    if multi_task:
        output_return = Dense(1, name="output_return")(embedding)
        output_spread = Dense(1, name="output_spread")(embedding)
        model = Model(inputs, [output_return, output_spread])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss={
                "output_return": "huber",
                "output_spread": "huber"
            },
            loss_weights={
                "output_return": 1.0,
                "output_spread": 0.3
            }
        )
    else:
        output_return = Dense(1, name="output_return")(embedding)
        model = Model(inputs, output_return)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="huber")
        
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
    sample_shape = (45, 16)
    
    
    trans_m = build_transformer(sample_shape)
    print(f"Transformer model built successfully! Output shape: {trans_m.output_shape}")