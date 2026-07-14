import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

# Helper dynamic slice indices based on feature count to support feature pruning
def get_slice_indices(name, num_features):
    if num_features == 22:
        if 'slice_price' in name:
            return 0, 7
        elif 'slice_volume' in name:
            return 7, 11
        elif 'slice_tech' in name:
            return 11, 18
        elif 'slice_flow_div' in name:
            return 18, 22
    elif num_features == 42:
        if 'slice_price' in name:
            return 0, 12
        elif 'slice_volume' in name:
            return 12, 18
        elif 'slice_tech' in name:
            return 18, 34
        elif 'slice_flow_div' in name:
            return 34, 42
    elif num_features == 40:
        if 'slice_price' in name:
            return 0, 12
        elif 'slice_volume' in name:
            return 12, 18
        elif 'slice_tech' in name:
            return 18, 35
        elif 'slice_flow_div' in name:
            return 35, 40
    else: # 44 or default
        if 'slice_price' in name:
            return 0, 12
        elif 'slice_volume' in name:
            return 12, 18
        elif 'slice_tech' in name:
            return 18, 36
        elif 'slice_flow_div' in name:
            return 36, 44
    return None, None

# Thay thế tầng Lambda để tránh việc Keras 3 gọi eval/compile mã bytecode của python lambda,
# giúp bypass hoàn toàn lỗi SystemError: no locals found khi load_model trên Python 3.12+.
@tf.keras.utils.register_keras_serializable()
class CustomLambda(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        for key in ['function', 'function_type', 'output_shape', 'module']:
            kwargs.pop(key, None)
        super().__init__(**kwargs)

    def call(self, inputs):
        num_features = inputs.shape[-1]
        if num_features is None:
            num_features = 22
        name = getattr(self, 'name', '')
        start, end = get_slice_indices(name, num_features)
        if start is not None:
            return inputs[:, :, start:end]
        return inputs

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 1:
            input_shape = input_shape[0]
        num_features = input_shape[-1]
        if num_features is None:
            num_features = 22
        name = getattr(self, 'name', '')
        start, end = get_slice_indices(name, num_features)
        if start is not None:
            return (input_shape[0], input_shape[1], end - start)
        return input_shape

    def get_config(self):
        return super().get_config()

# Đăng ký đè tầng Lambda toàn cục trong Keras
tf.keras.utils.get_custom_objects()['Lambda'] = CustomLambda

# Monkeypatch built-in tf.keras.layers.Lambda to handle dynamic shapes in Keras 3 load_model
original_lambda_call = tf.keras.layers.Lambda.call
original_lambda_compute_output_shape = tf.keras.layers.Lambda.compute_output_shape

def patched_lambda_call(self, inputs, *args, **kwargs):
    name = getattr(self, 'name', '')
    if name and any(slice_name in name for slice_name in ['slice_price', 'slice_volume', 'slice_tech', 'slice_flow_div']):
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        num_features = inputs.shape[-1]
        if num_features is None:
            num_features = 22
        start, end = get_slice_indices(name, num_features)
        if start is not None:
            return inputs[:, :, start:end]
    return original_lambda_call(self, inputs, *args, **kwargs)

def patched_lambda_compute_output_shape(self, input_shape):
    name = getattr(self, 'name', '')
    if name and any(slice_name in name for slice_name in ['slice_price', 'slice_volume', 'slice_tech', 'slice_flow_div']):
        if isinstance(input_shape, list) and len(input_shape) == 1:
            input_shape = input_shape[0]
        num_features = input_shape[-1]
        if num_features is None:
            num_features = 22
        start, end = get_slice_indices(name, num_features)
        if start is not None:
            return (input_shape[0], input_shape[1], end - start)
    return original_lambda_compute_output_shape(self, input_shape)

tf.keras.layers.Lambda.call = patched_lambda_call
tf.keras.layers.Lambda.compute_output_shape = patched_lambda_compute_output_shape

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, LayerNormalization,
    Conv1D, Bidirectional, GRU, Multiply, Concatenate, GaussianNoise
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import numpy as np

class DualEngineXGBoost:
    def __init__(self):
        self.cls = XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.02, 
            random_state=42, eval_metric='logloss', n_jobs=1
        )
        self.reg = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.015, 
            random_state=42, eval_metric='rmse', n_jobs=1
        )
        self.model_cls = None
        self.model_reg = None
        
    def fit(self, X, y):
        # Bước 1: Tạo nhãn phân loại (1 nếu lợi nhuận >= 0, ngược lại 0)
        y_cls = (y >= 0).astype(int)
        
        if y.ndim > 1 and y.shape[1] > 1:
            self.model_cls = MultiOutputClassifier(self.cls)
            self.model_reg = MultiOutputRegressor(self.reg)
        else:
            y = y.ravel()
            y_cls = y_cls.ravel()
            self.model_cls = self.cls
            self.model_reg = self.reg
            
        print("   [Dual-Engine] Đang huấn luyện Engine 1 (Phân loại xu hướng)...")
        self.model_cls.fit(X, y_cls)
        
        print("   [Dual-Engine] Đang huấn luyện Engine 2 (Hồi quy biên độ)...")
        self.model_reg.fit(X, y)
        return self

    def predict(self, X):
        trend = self.model_cls.predict(X)
        magnitude = self.model_reg.predict(X)
        # Kích hoạt Hurdle Model: Chỉ giữ lại biên độ nếu Engine 1 dự báo tăng (1)
        return magnitude * trend

def build_xgboost_optimized(X_train_flat, y_train):
    print("⏳ Khởi tạo kiến trúc Dual-Engine (Classifier + Regressor)...")
    model = DualEngineXGBoost()
    model.fit(X_train_flat, y_train)
    return model


# ── PositionalEmbedding ──────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    """Gán thẻ thứ tự thời gian — weights trong build() chuẩn Keras 3."""

    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(self.sequence_length, self.d_model),
            initializer='uniform',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_embedding[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
        })
        return config


# ── TimeDecayAttention ───────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class TimeDecayAttention(tf.keras.layers.Layer):
    """Attention với hệ số suy giảm thời gian trainable per-head."""

    def __init__(self, num_heads, key_dim, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads    = num_heads
        self.key_dim      = key_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.query_dense = Dense(self.num_heads * self.key_dim)
        self.key_dense   = Dense(self.num_heads * self.key_dim)
        self.value_dense = Dense(self.num_heads * self.key_dim)
        self.out_dense   = Dense(d_model)
        self.log_gamma   = self.add_weight(
            name='log_gamma',
            shape=(self.num_heads, 1, 1),
            initializer='zeros',
            trainable=True,
        )
        self.dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len    = tf.shape(inputs)[1]

        Q = tf.reshape(self.query_dense(inputs),
                       (batch_size, seq_len, self.num_heads, self.key_dim))
        K = tf.reshape(self.key_dense(inputs),
                       (batch_size, seq_len, self.num_heads, self.key_dim))
        V = tf.reshape(self.value_dense(inputs),
                       (batch_size, seq_len, self.num_heads, self.key_dim))

        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        score = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.key_dim, tf.float32)
        )
        positions = tf.cast(tf.range(seq_len), tf.float32)
        delta_t   = tf.abs(positions[:, None] - positions[None, :])
        delta_t   = delta_t[None, None, :, :]
        gamma     = tf.math.softplus(self.log_gamma)
        gamma     = gamma[None, :, :, :]
        score     = score - gamma * delta_t

        attn_weights = tf.nn.softmax(score, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        out = tf.matmul(attn_weights, V)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.num_heads * self.key_dim))
        return self.out_dense(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads':    self.num_heads,
            'key_dim':      self.key_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


# ── UncertaintyWeightsLayer ──────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class UncertaintyWeightsLayer(tf.keras.layers.Layer):
    """Lưu log-variances cho Kendall uncertainty weighting."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.log_var1 = self.add_weight(
            name='log_var_return', shape=(),
            initializer='zeros', trainable=True,
        )
        self.log_var2 = self.add_weight(
            name='log_var_spread', shape=(),
            initializer='zeros', trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs

    def get_config(self):
        return super().get_config()


# ── MultiTaskModel ───────────────────────────────────────────────────
# THIẾT KẾ MỚI:
# - Kế thừa tf.keras.Model (subclass API) — có call() đầy đủ
# - KHÔNG dùng Functional constructor (inputs=, outputs=)
# - Lưu/load bằng save_weights / load_weights thay vì model.save()
# - build_transformer() trả về tuple (multitask_model, base_functional_model)
#   để backtest dùng base_functional_model.inputs an toàn

@tf.keras.utils.register_keras_serializable()
class MultiTaskModel(tf.keras.Model):
    """
    Wrapper subclass model cho multi-task training.
    Bên trong chứa một Functional model (backbone) làm feature extractor.
    """

    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def get_layer(self, name=None, index=None):
        # Delegate get_layer về backbone để feature extractor hoạt động
        return self.backbone.get_layer(name=name, index=index)

    @property
    def inputs(self):
        return self.backbone.inputs

    @property
    def input(self):
        return self.backbone.input

    def get_config(self):
        config = super().get_config()
        config.update({
            'backbone': tf.keras.utils.serialize_keras_object(self.backbone),
        })
        return config

    @classmethod
    def from_config(cls, config):
        backbone = tf.keras.utils.deserialize_keras_object(config.pop('backbone'))
        return cls(backbone=backbone, **config)

    def train_step(self, data):
        x, y = data
        y_return = y["output_return"]
        y_spread = y["output_spread"]

        w_layer  = self.backbone.get_layer("uncertainty_weights")
        log_var1 = w_layer.log_var1
        log_var2 = w_layer.log_var2

        with tf.GradientTape() as tape:
            pred_return, pred_spread = self(x, training=True)
            if pred_return.shape[-1] == 1:
                pred_return = tf.squeeze(pred_return, axis=-1)
            if pred_spread.shape[-1] == 1:
                pred_spread = tf.squeeze(pred_spread, axis=-1)

            loss_r = tf.keras.losses.huber(y_return, pred_return)
            loss_s = tf.keras.losses.huber(y_spread, pred_spread)

            loss_r_w = tf.exp(-log_var1) * loss_r + 0.5 * log_var1
            loss_s_w = tf.exp(-log_var2) * loss_s + 0.5 * log_var2
            total    = tf.reduce_mean(loss_r_w + loss_s_w)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss":         total,
            "loss_return":  tf.reduce_mean(loss_r),
            "loss_spread":  tf.reduce_mean(loss_s),
            "sigma_return": tf.exp(0.5 * log_var1),
            "sigma_spread": tf.exp(0.5 * log_var2),
        }

    def test_step(self, data):
        x, y = data
        y_return = y["output_return"]
        y_spread = y["output_spread"]

        w_layer  = self.backbone.get_layer("uncertainty_weights")
        log_var1 = w_layer.log_var1
        log_var2 = w_layer.log_var2

        pred_return, pred_spread = self(x, training=False)
        if pred_return.shape[-1] == 1:
            pred_return = tf.squeeze(pred_return, axis=-1)
        if pred_spread.shape[-1] == 1:
            pred_spread = tf.squeeze(pred_spread, axis=-1)

        loss_r = tf.keras.losses.huber(y_return, pred_return)
        loss_s = tf.keras.losses.huber(y_spread, pred_spread)

        loss_r_w = tf.exp(-log_var1) * loss_r + 0.5 * log_var1
        loss_s_w = tf.exp(-log_var2) * loss_s + 0.5 * log_var2
        total    = tf.reduce_mean(loss_r_w + loss_s_w)

        return {
            "loss":        total,
            "loss_return": tf.reduce_mean(loss_r),
            "loss_spread": tf.reduce_mean(loss_s),
        }


# Custom helper removal (Standard tf.keras serialization used instead)


# ── GLU helper ───────────────────────────────────────────────────────
def glu(x, d_model):
    gate   = Dense(d_model, activation='sigmoid',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    linear = Dense(d_model,
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return Multiply()([linear, gate])


# ── build_transformer ────────────────────────────────────────────────
def build_transformer(input_shape, d_model=128, num_heads=8, key_dim=16, dropout_rate=0.3,
                      learning_rate=1e-4, multi_task=True):
    """
    Trả về:
      - multi_task=True : MultiTaskModel (subclass) bọc backbone Functional
      - multi_task=False: Functional model thông thường
    """
    inputs = Input(shape=input_shape)
    # Giai đoạn 1: Noise Injection vào đầu vào để tránh học vẹt
    x_noise = GaussianNoise(0.02, name="input_noise")(inputs)
    num_features = input_shape[-1]
    if num_features is None:
        num_features = 44

    if num_features == 22:
        x_price  = tf.keras.layers.Lambda(lambda x: x[:, :,  0:7],  output_shape=lambda s: (s[0], s[1], 7),  name="slice_price")(x_noise)
        x_volume = tf.keras.layers.Lambda(lambda x: x[:, :,  7:11], output_shape=lambda s: (s[0], s[1], 4),  name="slice_volume")(x_noise)
        x_tech   = tf.keras.layers.Lambda(lambda x: x[:, :, 11:18], output_shape=lambda s: (s[0], s[1], 7),  name="slice_tech")(x_noise)
        x_flow_div = tf.keras.layers.Lambda(lambda x: x[:, :, 18:22], output_shape=lambda s: (s[0], s[1], 4), name="slice_flow_div")(x_noise)
    elif num_features == 42:
        x_price  = tf.keras.layers.Lambda(lambda x: x[:, :,  0:12], output_shape=lambda s: (s[0], s[1], 12), name="slice_price")(x_noise)
        x_volume = tf.keras.layers.Lambda(lambda x: x[:, :, 12:18], output_shape=lambda s: (s[0], s[1], 6), name="slice_volume")(x_noise)
        x_tech   = tf.keras.layers.Lambda(lambda x: x[:, :, 18:34], output_shape=lambda s: (s[0], s[1], 16), name="slice_tech")(x_noise)
        x_flow_div = tf.keras.layers.Lambda(lambda x: x[:, :, 34:42], output_shape=lambda s: (s[0], s[1], 8), name="slice_flow_div")(x_noise)
    elif num_features == 40:
        x_price  = tf.keras.layers.Lambda(lambda x: x[:, :,  0:12], output_shape=lambda s: (s[0], s[1], 12), name="slice_price")(x_noise)
        x_volume = tf.keras.layers.Lambda(lambda x: x[:, :, 12:18], output_shape=lambda s: (s[0], s[1], 6), name="slice_volume")(x_noise)
        x_tech   = tf.keras.layers.Lambda(lambda x: x[:, :, 18:35], output_shape=lambda s: (s[0], s[1], 17), name="slice_tech")(x_noise)
        x_flow_div = tf.keras.layers.Lambda(lambda x: x[:, :, 35:40], output_shape=lambda s: (s[0], s[1], 5), name="slice_flow_div")(x_noise)
    else:
        x_price  = tf.keras.layers.Lambda(lambda x: x[:, :,  0:12], output_shape=lambda s: (s[0], s[1], 12), name="slice_price")(x_noise)
        x_volume = tf.keras.layers.Lambda(lambda x: x[:, :, 12:18], output_shape=lambda s: (s[0], s[1], 6), name="slice_volume")(x_noise)
        x_tech   = tf.keras.layers.Lambda(lambda x: x[:, :, 18:36], output_shape=lambda s: (s[0], s[1], 18), name="slice_tech")(x_noise)
        x_flow_div = tf.keras.layers.Lambda(lambda x: x[:, :, 36:44], output_shape=lambda s: (s[0], s[1], 8), name="slice_flow_div")(x_noise)

    def branch(x, filters, num_heads_branch, key_dim_branch, gru_units, latent_dim, name):
        x = Conv1D(filters, 3, padding='same', activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = glu(x, filters)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = PositionalEmbedding(sequence_length=input_shape[0],
                                d_model=filters)(x)
        a = TimeDecayAttention(num_heads=num_heads_branch, key_dim=key_dim_branch,
                               dropout_rate=dropout_rate)(x)
        x = LayerNormalization(epsilon=1e-6)(x + a)
        x = Bidirectional(GRU(gru_units, return_sequences=False,
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(x)
        return Dense(latent_dim, activation='relu', name=name,
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    if num_features == 22:
        p_emb = branch(x_price,  32, num_heads, key_dim, 12, 12, "latent_price")
        v_emb = branch(x_volume, 16, num_heads, key_dim,  6,  6, "latent_volume")
        t_emb = branch(x_tech,   16, num_heads, key_dim,  6,  6, "latent_tech")
        f_emb = branch(x_flow_div, 16, num_heads, key_dim,  6,  6, "latent_flow_div")
    else:
        p_emb = branch(x_price,  64, num_heads, key_dim, 16, 16, "latent_price")
        v_emb = branch(x_volume, 32, num_heads, key_dim,  8,  8, "latent_volume")
        t_emb = branch(x_tech,   32, num_heads, key_dim,  8,  8, "latent_tech")
        f_emb = branch(x_flow_div, 32, num_heads, key_dim,  8,  8, "latent_flow_div")

    embedding = Concatenate(name="latent_embedding")([p_emb, v_emb, t_emb, f_emb])

    if multi_task:
        ew    = UncertaintyWeightsLayer(name="uncertainty_weights")(embedding)
        out_r = Dense(3, name="output_return", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ew)
        out_s = Dense(3, name="output_spread", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ew)

        # Backbone là Functional model thuần — serialize được
        backbone = Model(inputs=inputs, outputs=[out_r, out_s], name="backbone")

        # Wrapper subclass — dùng để train với custom train_step
        model = MultiTaskModel(backbone=backbone, name="multi_task_model")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

        # Build ngay để .inputs hoạt động
        dummy = np.zeros((1,) + input_shape, dtype=np.float32)
        model(dummy, training=False)
        return model
    else:
        out_r = Dense(3, name="output_return", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(embedding)
        model = Model(inputs=inputs, outputs=out_r)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='huber',
        )
        return model


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("=== TESTING AI MODEL ===")
    m = build_transformer((45, 44))
    print("outputs:", [o.name for o in m.backbone.outputs])
    print("inputs :", m.inputs)
    
    dummy = np.zeros((2, 45, 44), dtype=np.float32)
    out = m(dummy, training=False)
    print("Test forward OK, output shapes:", [o.shape for o in out])