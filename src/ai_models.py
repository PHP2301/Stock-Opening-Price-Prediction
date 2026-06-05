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
# 2. MÔ HÌNH TRANSFORMER PHÂN NHÁNH & HỌC ĐA NHIỆM (3 Branches, Uncertainty Weighting Loss)
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


@tf.keras.utils.register_keras_serializable()
class TimeDecayAttention(tf.keras.layers.Layer):
    """
    Cơ chế Attention trễ thời gian (Time-Decay Attention) tự học hệ số suy giảm gamma 
    cho từng Head để giảm mức độ chú ý vào các phiên quá xa trong quá khứ.
    """
    def __init__(self, num_heads, key_dim, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        d_model = input_shape[-1]
        self.query_dense = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.key_dense = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.value_dense = tf.keras.layers.Dense(self.num_heads * self.key_dim)
        self.out_dense = tf.keras.layers.Dense(d_model)
        
        # Hệ số decay log_gamma của từng head (đảm bảo gamma > 0 qua Softplus)
        self.log_gamma = self.add_weight(
            name='log_gamma',
            shape=(self.num_heads, 1, 1),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        Q = self.query_dense(inputs)
        K = self.key_dense(inputs)
        V = self.value_dense(inputs)
        
        # Reshape sang (batch, seq, heads, dim)
        Q = tf.reshape(Q, (batch_size, seq_len, self.num_heads, self.key_dim))
        K = tf.reshape(K, (batch_size, seq_len, self.num_heads, self.key_dim))
        V = tf.reshape(V, (batch_size, seq_len, self.num_heads, self.key_dim))
        
        # Transpose sang (batch, heads, seq, dim)
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])
        
        # Tích vô hướng scaled QK^T
        score = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        
        # Tính khoảng cách delta_t giữa các phiên: |i - j|
        positions = tf.cast(tf.range(seq_len), tf.float32)
        delta_t = tf.abs(positions[:, None] - positions[None, :])
        delta_t = tf.expand_dims(tf.expand_dims(delta_t, 0), 0) # (1, 1, seq, seq)
        
        gamma = tf.math.softplus(self.log_gamma) # (heads, 1, 1)
        gamma = tf.expand_dims(gamma, 0) # (1, heads, 1, 1)
        
        decay = gamma * delta_t
        score = score - decay  # Áp dụng suy giảm theo khoảng cách thời gian
        
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        out = tf.matmul(attention_weights, V)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.num_heads * self.key_dim))
        return self.out_dense(out)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class UncertaintyWeightsLayer(tf.keras.layers.Layer):
    """
    Lớp lưu trữ trọng số bất định (trainable log-variances) cho loss đa nhiệm.
    Giúp tránh lỗi Keras 3 serialization khi gán biến trực tiếp vào Functional Model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_var1 = self.add_weight(name="log_var_return", shape=(), initializer="zeros", trainable=True)
        self.log_var2 = self.add_weight(name="log_var_spread", shape=(), initializer="zeros", trainable=True)
        
    def call(self, inputs):
        return inputs
        
    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class MultiTaskModel(tf.keras.models.Model):
    """
    Mô hình học đa nhiệm tự động cân bằng trọng số tổn thất 
    (Kendall 2018 Uncertainty Weighting Loss) cho lợi nhuận (Return) và độ rộng giá (Spread).
    """
    def get_config(self):
        from keras.src.models.functional import Functional
        return Functional.get_config(self)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.src.models.functional import Functional
        model = Functional.from_config(config, custom_objects=custom_objects)
        model.__class__ = cls
        return model
        
    def train_step(self, data):
        x, y = data
        y_return = y["output_return"]
        y_spread = y["output_spread"]
        
        # Lấy weights từ UncertaintyWeightsLayer
        w_layer = self.get_layer("uncertainty_weights")
        log_var1 = w_layer.log_var1
        log_var2 = w_layer.log_var2
        
        with tf.GradientTape() as tape:
            pred_return, pred_spread = self(x, training=True)
            
            # Loss cơ bản sử dụng Huber Loss
            loss_return = tf.keras.losses.huber(y_return, pred_return)
            loss_spread = tf.keras.losses.huber(y_spread, pred_spread)
            
            # Tổn thất tích hợp trọng số bất định
            loss_return_weighted = tf.exp(-log_var1) * loss_return + 0.5 * log_var1
            loss_spread_weighted = tf.exp(-log_var2) * loss_spread + 0.5 * log_var2
            
            total_loss = tf.reduce_mean(loss_return_weighted + loss_spread_weighted)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {
            "loss": total_loss,
            "loss_return": tf.reduce_mean(loss_return),
            "loss_spread": tf.reduce_mean(loss_spread),
            "sigma_return": tf.exp(0.5 * log_var1),
            "sigma_spread": tf.exp(0.5 * log_var2)
        }
        
    def test_step(self, data):
        x, y = data
        y_return = y["output_return"]
        y_spread = y["output_spread"]
        
        # Lấy weights từ UncertaintyWeightsLayer
        w_layer = self.get_layer("uncertainty_weights")
        log_var1 = w_layer.log_var1
        log_var2 = w_layer.log_var2
        
        pred_return, pred_spread = self(x, training=False)
        
        # Loss cơ bản sử dụng Huber Loss
        loss_return = tf.keras.losses.huber(y_return, pred_return)
        loss_spread = tf.keras.losses.huber(y_spread, pred_spread)
        
        # Tổn thất tích hợp trọng số bất định
        loss_return_weighted = tf.exp(-log_var1) * loss_return + 0.5 * log_var1
        loss_spread_weighted = tf.exp(-log_var2) * loss_spread + 0.5 * log_var2
        
        total_loss = tf.reduce_mean(loss_return_weighted + loss_spread_weighted)
        
        return {
            "loss": total_loss,
            "loss_return": tf.reduce_mean(loss_return),
            "loss_spread": tf.reduce_mean(loss_spread)
        }


def glu(x, d_model):
    gate = Dense(d_model, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    linear = Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return Multiply()([linear, gate])


def build_transformer(input_shape, d_model=128, dropout_rate=0.3, learning_rate=1e-4, multi_task=True):
    """
    Xây dựng kiến trúc Transformer Phân Nhánh để học các nhóm đặc trưng độc lập:
    - Nhánh 1 (Giá & Động lượng): 12 đặc trưng
    - Nhánh 2 (Khối lượng & Biến động): 6 đặc trưng
    - Nhánh 3 (Kỹ thuật, Vĩ mô & Lịch): 16 đặc trưng
    """
    inputs = Input(shape=input_shape)
    
    # Chia tách nhánh bằng Lambda layers để đảm bảo tương thích serialization
    x_price = tf.keras.layers.Lambda(lambda x: x[:, :, 0:12], name="slice_price")(inputs)
    x_volume = tf.keras.layers.Lambda(lambda x: x[:, :, 12:18], name="slice_volume")(inputs)
    x_tech = tf.keras.layers.Lambda(lambda x: x[:, :, 18:34], name="slice_tech")(inputs)
    
    # --- NHÁNH 1: GIÁ & ĐỘNG LƯỢNG (12 feats -> d=64) ---
    p = Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x_price)
    p = LayerNormalization(epsilon=1e-6)(p)
    p = glu(p, 64)
    p = LayerNormalization(epsilon=1e-6)(p)
    p = PositionalEmbedding(sequence_length=input_shape[0], d_model=64)(p)
    p_attn = TimeDecayAttention(num_heads=4, key_dim=16, dropout_rate=dropout_rate)(p)
    p = LayerNormalization(epsilon=1e-6)(p + p_attn)
    p_out = Bidirectional(GRU(16, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(p)
    p_emb = Dense(16, activation="relu", name="latent_price", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(p_out)
    
    # --- NHÁNH 2: KHỐI LƯỢNG & BIẾN ĐỘNG (6 feats -> d=32) ---
    v = Conv1D(32, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x_volume)
    v = LayerNormalization(epsilon=1e-6)(v)
    v = glu(v, 32)
    v = LayerNormalization(epsilon=1e-6)(v)
    v = PositionalEmbedding(sequence_length=input_shape[0], d_model=32)(v)
    v_attn = TimeDecayAttention(num_heads=2, key_dim=16, dropout_rate=dropout_rate)(v)
    v = LayerNormalization(epsilon=1e-6)(v + v_attn)
    v_out = Bidirectional(GRU(8, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(v)
    v_emb = Dense(8, activation="relu", name="latent_volume", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(v_out)
    
    # --- NHÁNH 3: KỸ THUẬT, VĨ MÔ & LỊCH (16 feats -> d=32) ---
    t = Conv1D(32, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x_tech)
    t = LayerNormalization(epsilon=1e-6)(t)
    t = glu(t, 32)
    t = LayerNormalization(epsilon=1e-6)(t)
    t = PositionalEmbedding(sequence_length=input_shape[0], d_model=32)(t)
    t_attn = TimeDecayAttention(num_heads=2, key_dim=16, dropout_rate=dropout_rate)(t)
    t = LayerNormalization(epsilon=1e-6)(t + t_attn)
    t_out = Bidirectional(GRU(8, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(t)
    t_emb = Dense(8, activation="relu", name="latent_tech", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(t_out)
    
    # Ghép 3 không gian ẩn thành Bottleneck 32 chiều
    embedding = tf.keras.layers.Concatenate(name="latent_embedding")([p_emb, v_emb, t_emb])
    
    if multi_task:
        # Nhúng layer lưu weights bất định vào graph để được tự động serialize
        embedding_weighted = UncertaintyWeightsLayer(name="uncertainty_weights")(embedding)
        output_return = Dense(1, name="output_return")(embedding_weighted)
        output_spread = Dense(1, name="output_spread")(embedding_weighted)
        model = MultiTaskModel(inputs=inputs, outputs=[output_return, output_spread])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer)
    else:
        output_return = Dense(1, name="output_return")(embedding)
        model = Model(inputs=inputs, outputs=output_return)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="huber")
        
    return model


if __name__ == "__main__":
    import sys
    import os
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    print("=== TESTING AI MODEL ARCHITECTURES ===")
    sample_shape = (45, 34)
    trans_m = build_transformer(sample_shape)
    print(f"Transformer model built successfully! Output shapes: {[o.name for o in trans_m.outputs]}")