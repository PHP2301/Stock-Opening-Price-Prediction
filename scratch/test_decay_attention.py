import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import numpy as np

@tf.keras.utils.register_keras_serializable()
class TimeDecayAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = d_model // num_heads
        
        self.q_proj = Dense(d_model)
        self.k_proj = Dense(d_model)
        self.v_proj = Dense(d_model)
        self.out_proj = Dense(d_model)
        self.dropout = Dropout(dropout_rate)
        
    def build(self, input_shape):
        self.log_gamma = self.add_weight(
            name="log_gamma",
            shape=(self.num_heads, 1, 1),
            initializer=tf.keras.initializers.Constant(-2.0),
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        
        q = tf.transpose(tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)), [0, 2, 1, 3])
        
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        r = tf.range(seq_len, dtype=tf.float32)
        dist = tf.abs(r[:, None] - r[None, :])
        
        gamma = tf.exp(self.log_gamma)
        penalty = - gamma * dist[None, :, :]
        scores = scores + penalty[None, :, :, :]
        
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        
        context = tf.matmul(attn_weights, v)
        context = tf.reshape(tf.transpose(context, [0, 2, 1, 3]), (batch_size, seq_len, self.d_model))
        out = self.out_proj(context)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config

# Test compilation and gradient existence
inputs = Input(shape=(45, 128))
layer = TimeDecayAttention(d_model=128, num_heads=8)
outputs = layer(inputs)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Dummy data
X_dummy = np.random.randn(32, 45, 128)
y_dummy = np.random.randn(32, 45, 128)

print("Compiling model: SUCCESS")
loss_before = model.evaluate(X_dummy, y_dummy, verbose=0)
model.train_on_batch(X_dummy, y_dummy)
loss_after = model.evaluate(X_dummy, y_dummy, verbose=0)
print("Training 1 batch: SUCCESS")
print("Loss before:", loss_before, "Loss after:", loss_after)
print("log_gamma:", layer.log_gamma.numpy().ravel())
