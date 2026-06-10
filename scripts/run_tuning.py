import os, sys, random, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data
from src.features import DataTransformer
from src.ai_models import build_transformer

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

X_train = X_val = y_train = y_val = y_train_spread = y_val_spread = None


def objective(trial):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    d_model      = trial.suggest_categorical('d_model',    [64, 128, 256])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    dropout_rate = trial.suggest_float('dropout_rate',     0.1, 0.4)
    learning_rate= trial.suggest_float('learning_rate',    1e-5, 5e-4, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [32, 64])

    # FIX: multi_task=True để hyperparams khớp với kiến trúc train thật
    model = build_transformer(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    d_model=d_model,
    heads=num_heads,        # ← đổi tên biến
    dropout_rate=dropout_rate,
    learning_rate=learning_rate,
    multi_task=True,
        )

    early_stop = EarlyStopping(
        monitor='val_loss', patience=7,       # tăng từ 5→7 cho multi-task ổn định hơn
        restore_best_weights=True, verbose=0,
    )

    # FIX: truyền đủ cả 2 targets cho multi-task
    history = model.fit(
        X_train,
        {"output_return": y_train, "output_spread": y_train_spread},
        validation_data=(
            X_val,
            {"output_return": y_val, "output_spread": y_val_spread},
        ),
        epochs=20, batch_size=batch_size,
        callbacks=[early_stop], verbose=0,
    )

    return min(history.history['val_loss'])


if __name__ == "__main__":
    TICKERS = ["VNM.VN", "GOOGL", "META"]

    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
            print(f"🎯 Tuning cho: {TICKERS[0]}")
        elif arg == "ALL":
            print(f"🎯 Tuning cho toàn bộ: {TICKERS}")
    else:
        print(f"🎯 Tuning tuần tự cho: {TICKERS}")

    for t_ticker in TICKERS:
        print(f"\n{'='*50}\n🚀 [OPTUNA] {t_ticker}\n{'='*50}")

        df = fetch_and_prepare_data(t_ticker, "2022-01-01", "2026-05-20")

        transformer = DataTransformer(time_steps=45)
        X_sc, y_sc, y_sp_sc = transformer.fit_transform_train_only(df, train_ratio=0.8)
        X_3D, y_3D, y_sp_3D = transformer.create_sliding_windows(X_sc, y_sc, y_sp_sc)

        X_tr_all, y_tr_all, _, _, _, y_sp_tr_all, _ = \
            transformer.split_train_test_chronological(df, X_3D, y_3D, y_sp_3D, train_ratio=0.8)

        # FIX: Purge Gap 45 phiên giữa train và val
        val_size = int(len(X_tr_all) * 0.1)
        purge    = 45
        train_end = len(X_tr_all) - val_size - purge

        X_train        = X_tr_all[:train_end]
        y_train        = y_tr_all[:train_end].ravel()
        y_train_spread = y_sp_tr_all[:train_end].ravel() if y_sp_tr_all is not None else np.zeros(train_end)

        X_val          = X_tr_all[-val_size:]
        y_val          = y_tr_all[-val_size:].ravel()
        y_val_spread   = y_sp_tr_all[-val_size:].ravel() if y_sp_tr_all is not None else np.zeros(val_size)

        print(f"📊 Train={X_train.shape[0]}, Val={X_val.shape[0]} "
              f"(Purge gap={purge} giữa train/val)")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25)

        print(f"\n🏆 Hoàn tất {t_ticker}! Val Loss = {study.best_value:.6f}")
        for k, v in study.best_params.items():
            print(f"   🔹 {k}: {v}")

        config_dir = os.path.join(ROOT_DIR, 'config')
        os.makedirs(config_dir, exist_ok=True)
        out_path = os.path.join(config_dir, f'best_transformer_params_{t_ticker}.json')
        with open(out_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"💾 Lưu config: {out_path}")

