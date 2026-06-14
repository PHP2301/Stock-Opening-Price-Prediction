import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, LayerNormalization,
    Conv1D, Bidirectional, GRU, Multiply, Concatenate
)
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


def build_xgboost_optimized(X_train_flat, y_train):
    xgb_model = XGBRegressor(random_state=42, n_jobs=1)
    param_distributions = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=6, cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=1, random_state=42,
    )
    print("⏳ Đang quét nhanh bộ tham số tối ưu cho XGBoost...")
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        search.fit(X_train_flat, y_train)
    else:
        search.fit(X_train_flat, y_train.ravel())
    print(f"🔥 Tham số tối ưu: {search.best_params_}")
    return search.best_estimator_


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

    x_price  = tf.keras.layers.Lambda(lambda x: x[:, :,  0:12], name="slice_price")(inputs)
    x_volume = tf.keras.layers.Lambda(lambda x: x[:, :, 12:18], name="slice_volume")(inputs)
    x_tech   = tf.keras.layers.Lambda(lambda x: x[:, :, 18:34], name="slice_tech")(inputs)
    x_flow_div = tf.keras.layers.Lambda(lambda x: x[:, :, 34:42], name="slice_flow_div")(inputs)

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

    p_emb = branch(x_price,  64, num_heads, key_dim, 16, 16, "latent_price")
    v_emb = branch(x_volume, 32, num_heads, key_dim,  8,  8, "latent_volume")
    t_emb = branch(x_tech,   32, num_heads, key_dim,  8,  8, "latent_tech")
    f_emb = branch(x_flow_div, 32, num_heads, key_dim,  8,  8, "latent_flow_div")

    embedding = Concatenate(name="latent_embedding")([p_emb, v_emb, t_emb, f_emb])

    if multi_task:
        ew    = UncertaintyWeightsLayer(name="uncertainty_weights")(embedding)
        out_r = Dense(3, name="output_return")(ew)
        out_s = Dense(3, name="output_spread")(ew)

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
        out_r = Dense(3, name="output_return")(embedding)
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
    m = build_transformer((45, 42))
    print("outputs:", [o.name for o in m.backbone.outputs])
    print("inputs :", m.inputs)
    
    dummy = np.zeros((2, 45, 42), dtype=np.float32)
    out = m(dummy, training=False)
    print("Test forward OK, output shapes:", [o.shape for o in out])