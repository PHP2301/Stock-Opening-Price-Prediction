import os, sys, joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data
from src.features import DataTransformer
from src.ai_models import (
    PositionalEmbedding, TimeDecayAttention,
    MultiTaskModel, UncertaintyWeightsLayer, CustomLambda,
)

custom_objects = {
    'PositionalEmbedding':     PositionalEmbedding,
    'TimeDecayAttention':      TimeDecayAttention,
    'UncertaintyWeightsLayer': UncertaintyWeightsLayer,
    'MultiTaskModel':          MultiTaskModel,
    'Lambda':                  CustomLambda,
}

for ticker in ["META"]:
    print(f"\n🔬 Ticker: {ticker}")
    trans_path = os.path.join(ROOT_DIR, 'models', f'transformer_model_{ticker}.keras')
    if os.path.exists(trans_path):
        model = tf.keras.models.load_model(trans_path, custom_objects=custom_objects, safe_mode=False)
        print(f"   Model inputs: {model.inputs}")
    else:
        print("   Model not found")

    df = fetch_and_prepare_data(ticker, start_date="2020-01-01", end_date="2021-01-01")
    dt = DataTransformer(time_steps=45, num_features=42)
    feats = dt.transform_df(df)
    print(f"   DataTransformer feature cols count: {len(dt.feature_cols)}")
    print(f"   feats shape: {feats.shape}")
