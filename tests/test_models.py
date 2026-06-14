import unittest
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_models import build_transformer, PositionalEmbedding, TimeDecayAttention, MultiTaskModel, UncertaintyWeightsLayer

class TestAIModels(unittest.TestCase):
    def test_build_transformer_multitask(self):
        """Kiểm tra việc build mô hình MultiTask Transformer thành công"""
        input_shape = (45, 42)
        model = build_transformer(input_shape, multi_task=True)
        self.assertIsInstance(model, MultiTaskModel)
        self.assertEqual(len(model.backbone.outputs), 2)
        self.assertEqual(model.backbone.outputs[0].shape[-1], 3)
        self.assertEqual(model.backbone.outputs[1].shape[-1], 3)

    def test_build_transformer_single_task(self):
        """Kiểm tra việc build mô hình SingleTask Transformer thành công"""
        input_shape = (45, 42)
        model = build_transformer(input_shape, multi_task=False)
        self.assertNotIsInstance(model, MultiTaskModel)
        self.assertEqual(len(model.outputs), 1)
        self.assertEqual(model.outputs[0].shape[-1], 3)

    def test_time_decay_attention_forward(self):
        """Kiểm tra lớp TimeDecayAttention thực hiện phép biến đổi forward bình thường"""
        layer = TimeDecayAttention(num_heads=2, key_dim=8)
        inputs = tf.random.normal((4, 10, 16)) # batch=4, seq=10, dim=16
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, (4, 10, 16))

    def test_uncertainty_weights_layer_in_model(self):
        """Kiểm tra UncertaintyWeightsLayer được nhúng đúng vào kiến trúc MultiTask"""
        input_shape = (45, 42)
        model = build_transformer(input_shape, multi_task=True)
        layer_names = [layer.name for layer in model.backbone.layers]
        self.assertIn("uncertainty_weights", layer_names)
        w_layer = model.backbone.get_layer("uncertainty_weights")
        self.assertIsInstance(w_layer, UncertaintyWeightsLayer)
        # Verify both log-variance weights exist and are trainable
        self.assertEqual(len([w for w in w_layer.trainable_weights]), 2)

if __name__ == '__main__':
    unittest.main()
