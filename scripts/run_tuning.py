import os
import sys
import random
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import optuna

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data
from src.features import DataTransformer
from src.ai_models import build_transformer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ════════════════════════════════════════════════════════════════════
# Search space riêng cho từng ticker
# - VNM.VN: thị trường VN noise cao → model nhỏ tránh overfit
# - META:   đã overfit nặng với d_model=256 → cap tại 128
# - GOOGL:  đang tốt với d_model=64 → cho phép rộng hơn 1 chút
# ════════════════════════════════════════════════════════════════════
SEARCH_SPACE = {
    'VNM.VN': {'d_model': [32, 64],      'num_heads': [2, 4]},
    'GOOGL':  {'d_model': [64, 128],     'num_heads': [2, 4, 8]},
    'META':   {'d_model': [64, 128],     'num_heads': [2, 4]},
}

# Số trial động: META/VNM cần tìm nhiều hơn vì space mới
N_TRIALS = {
    'VNM.VN': 40,
    'GOOGL':  25,
    'META':   40,
}

# Epochs trong tuning tăng 20→40 để model lớn kịp hội tụ
# Tránh val_loss ảo thấp ở epoch 1-5 khiến Optuna chọn nhầm model lớn
TUNING_EPOCHS  = 40
TUNING_PATIENCE = 10

# Biến global chứa data cho objective (Optuna yêu cầu single-arg callable)
_X_train = _X_val = _y_train = _y_val = None
_y_train_spread = _y_val_spread = None
_current_ticker = None


def objective(trial):
    """
    Hàm mục tiêu Optuna — dùng biến global _current_ticker để
    load search space đúng per-ticker mà không cần thay đổi signature.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    ticker = _current_ticker
    space  = SEARCH_SPACE.get(ticker, SEARCH_SPACE['GOOGL'])

    # ── Gợi ý siêu tham số từ search space riêng của ticker ──────
    d_model       = trial.suggest_categorical('d_model',      space['d_model'])
    num_heads     = trial.suggest_categorical('num_heads',    space['num_heads'])
    dropout_rate  = trial.suggest_float('dropout_rate',       0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate',      1e-5, 5e-4, log=True)
    batch_size    = trial.suggest_categorical('batch_size',   [32, 64])

    # Kiểm tra tính hợp lệ: d_model phải chia hết cho num_heads
    if d_model % num_heads != 0:
        # Pruning trial không hợp lệ thay vì raise error
        raise optuna.exceptions.TrialPruned()

    model = build_transformer(
        input_shape=(_X_train.shape[1], _X_train.shape[2]),
        d_model=d_model,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        multi_task=True,
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=TUNING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        _X_train,
        {"output_return": _y_train, "output_spread": _y_train_spread},
        validation_data=(
            _X_val,
            {"output_return": _y_val, "output_spread": _y_val_spread},
        ),
        epochs=TUNING_EPOCHS,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    val_loss = min(history.history['val_loss'])

    # ════════════════════════════════════════════════════════════════
    # Complexity penalty: ưu tiên model nhỏ gọn nếu val_loss tương đương
    # Công thức: penalty tỷ lệ thuận với d_model
    # d_model=32  → penalty=0.00125
    # d_model=64  → penalty=0.0025
    # d_model=128 → penalty=0.005
    # d_model=256 → penalty=0.01  (không dùng nhưng để reference)
    # ════════════════════════════════════════════════════════════════
    complexity_penalty = (d_model / 256.0) * 0.01

    return val_loss + complexity_penalty


def run_tuning_for_ticker(t_ticker: str):
    global _X_train, _X_val, _y_train, _y_val
    global _y_train_spread, _y_val_spread, _current_ticker

    print(f"\n{'='*55}")
    print(f"🚀 [OPTUNA] {t_ticker}")
    print(f"   Search space : {SEARCH_SPACE.get(t_ticker, SEARCH_SPACE['GOOGL'])}")
    print(f"   N trials     : {N_TRIALS.get(t_ticker, 25)}")
    print(f"   Tuning epochs: {TUNING_EPOCHS} (patience={TUNING_PATIENCE})")
    print(f"{'='*55}")

    # ── Tải & chuẩn bị dữ liệu ────────────────────────────────────
    # Dùng 2022→nay để tuning nhanh (~1000 phiên, đủ đại diện)
    df = fetch_and_prepare_data(t_ticker, "2022-01-01", "2026-05-20")

    transformer = DataTransformer(time_steps=45)
    X_sc, y_sc, y_sp_sc = transformer.fit_transform_train_only(df, train_ratio=0.8)
    X_3D, y_3D, y_sp_3D = transformer.create_sliding_windows(X_sc, y_sc, y_sp_sc)

    X_tr_all, y_tr_all, _, _, _, y_sp_tr_all, _ = \
        transformer.split_train_test_chronological(
            df, X_3D, y_3D, y_sp_3D, train_ratio=0.8
        )

    # ── Val split với Purge Gap 45 phiên ──────────────────────────
    val_size  = int(len(X_tr_all) * 0.1)
    purge     = 45
    train_end = len(X_tr_all) - val_size - purge

    if train_end <= 0:
        print(f"⚠️  Không đủ data để tạo val split với purge={purge}. Bỏ qua purge.")
        train_end = len(X_tr_all) - val_size

    _X_train        = X_tr_all[:train_end]
    _y_train        = y_tr_all[:train_end].ravel()
    _y_train_spread = (y_sp_tr_all[:train_end].ravel()
                       if y_sp_tr_all is not None else np.zeros(train_end))

    _X_val          = X_tr_all[-val_size:]
    _y_val          = y_tr_all[-val_size:].ravel()
    _y_val_spread   = (y_sp_tr_all[-val_size:].ravel()
                       if y_sp_tr_all is not None else np.zeros(val_size))

    _current_ticker = t_ticker

    print(f"📊 Data: Train={_X_train.shape[0]}, Purge={purge}, Val={_X_val.shape[0]}")

    # ── Chạy Optuna ───────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Dùng TPESampler với seed cố định để tái lập kết quả
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study   = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS.get(t_ticker, 25),
        show_progress_bar=False,
    )

    # ── Kết quả ───────────────────────────────────────────────────
    best       = study.best_params
    best_loss  = study.best_value
    raw_loss   = best_loss - (best.get('d_model', 128) / 256.0) * 0.01

    print(f"\n🏆 {t_ticker} — Val Loss (với penalty) = {best_loss:.6f}")
    print(f"   Val Loss (thuần)                  = {raw_loss:.6f}")
    for k, v in best.items():
        print(f"   🔹 {k}: {v}")

    # ── Lưu config ────────────────────────────────────────────────
    config_dir = os.path.join(ROOT_DIR, 'config')
    os.makedirs(config_dir, exist_ok=True)
    out_path   = os.path.join(config_dir, f'best_transformer_params_{t_ticker}.json')

    # Đảm bảo key luôn là "num_heads" (không phải "heads")
    save_params = {
        'd_model':       best['d_model'],
        'num_heads':     best['num_heads'],
        'dropout_rate':  best['dropout_rate'],
        'learning_rate': best['learning_rate'],
        'batch_size':    best['batch_size'],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_params, f, indent=4)

    print(f"💾 Config lưu tại: {out_path}")

    # In top-3 trials để tham khảo
    print(f"\n📋 Top-3 trials tốt nhất:")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value
    )[:3]
    for rank, t in enumerate(sorted_trials, 1):
        penalty = (t.params.get('d_model', 128) / 256.0) * 0.01
        print(f"   #{rank}: val_loss={t.value:.6f} "
              f"(thuần={t.value - penalty:.6f}) "
              f"d_model={t.params.get('d_model')} "
              f"num_heads={t.params.get('num_heads')} "
              f"lr={t.params.get('learning_rate'):.2e}")

    return save_params


def main():
    TICKERS = ["VNM.VN", "GOOGL", "META"]

    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg == "ALL":
            pass  # chạy tất cả
        elif arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
            print(f"🎯 Tuning cho: {TICKERS[0]}")
        else:
            print(f"⚠️ Không nhận ra ticker '{sys.argv[1]}'. Chạy tất cả.")
    else:
        print(f"🎯 Tuning tuần tự cho: {TICKERS}")

    results = {}
    for t_ticker in TICKERS:
        try:
            params = run_tuning_for_ticker(t_ticker)
            results[t_ticker] = params
        except Exception as e:
            print(f"❌ Lỗi khi tuning {t_ticker}: {e}")
            import traceback
            traceback.print_exc()

    # ── Tóm tắt toàn bộ ──────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*55}")
        print(f"📊 TÓM TẮT TUNING")
        print(f"{'='*55}")
        for ticker, p in results.items():
            print(f"  {ticker:10s}: d_model={p['d_model']:3d}, "
                  f"num_heads={p['num_heads']}, "
                  f"lr={p['learning_rate']:.2e}, "
                  f"dropout={p['dropout_rate']:.2f}, "
                  f"batch={p['batch_size']}")

    print(f"\n✅ Tuning hoàn tất. Chạy training với params mới:")
    for ticker in results:
        print(f"   python scripts/run_training.py {ticker}")


if __name__ == "__main__":
    main()
