import os
import random
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf
import math
import joblib
import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from src.data_loader import fetch_and_prepare_data, format_vn, get_realtime_usd_vnd_rate
from src.features import DataTransformer
from src.ai_models import build_transformer

USD_TO_VND = get_realtime_usd_vnd_rate()
print(f"💵 [TỶ GIÁ] Sử dụng tỷ giá USD/VND realtime: {format_vn(USD_TO_VND)} VNĐ\n")

LOOKBACK_WINDOW = 45


def evaluate_predictions(y_true, y_pred, model_name, ticker):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    print(f"=== KẾT QUẢ MÔ HÌNH {model_name.upper()} ===")
    if "VNM" in ticker.upper():
        print(f"❌ Sai số RMSE: {format_vn(rmse)} VNĐ")
        print(f"🎯 Sai số MAE : {format_vn(mae)} VNĐ (Lệch trung bình: {mape:.2f}%)\n")
    else:
        rmse_usd = rmse / USD_TO_VND
        mae_usd  = mae  / USD_TO_VND
        print(f"❌ Sai số RMSE: {format_vn(rmse)} VNĐ (tương đương ${rmse_usd:.2f} USD)")
        print(f"🎯 Sai số MAE : {format_vn(mae)} VNĐ (tương đương ${mae_usd:.2f} USD - Lệch trung bình: {mape:.2f}%)\n")

    return rmse, mae, mape


def cosine_decay_with_warmup(epoch, target_lr=1e-4):
    warmup_epochs = 5
    total_epochs = 120
    min_lr = 1e-6
    if epoch < warmup_epochs:
        return min_lr + (target_lr - min_lr) * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (target_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    print("🚀 [HỆ THỐNG] Khởi động Pipeline Huấn luyện RIÊNG mô hình Transformer...")

    SENTIMENT_ENGINE = 'finbert'
    TICKERS    = ["VNM.VN", "GOOGL", "META"]
    START_TRAIN = "2010-01-01"
    END_PREDICT = "2026-05-20"

    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]

    callbacks = [
        EarlyStopping(monitor='loss', patience=15, min_delta=1e-5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        LearningRateScheduler(lambda epoch: cosine_decay_with_warmup(epoch, target_lr=learning_rate), verbose=0),
    ]

    models_dir  = os.path.join(ROOT_DIR, 'models')
    results_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'─'*50}")
        print(f"🔄 BẮT ĐẦU HUẤN LUYỆN TRANSFORMER CHO MÃ: {ticker}...")
        print(f"{'─'*50}")

        df = fetch_and_prepare_data(
            ticker, start_date=START_TRAIN, end_date=END_PREDICT,
            sentiment_engine=SENTIMENT_ENGINE,
        )

        dt = DataTransformer(time_steps=LOOKBACK_WINDOW)
        X_scaled, y_scaled, y_spread_scaled = dt.fit_transform_train_only(df, train_ratio=0.8)
        X_3D, y_3D, y_spread_3D = dt.create_sliding_windows(X_scaled, y_scaled, y_spread_scaled)

        X_train_i, y_train_i, X_test_i, y_test_i, y_test_raw_i, y_train_spread_i, y_test_spread_i = \
            dt.split_train_test_chronological(df, X_3D, y_3D, y_spread_3D, train_ratio=0.8)

        val_size = int(len(X_train_i) * 0.1)
        purge = 45
        if val_size > 0 and len(X_train_i) - val_size - purge > 0:
            train_end = len(X_train_i) - val_size - purge
            X_tr, y_tr = X_train_i[:train_end], y_train_i[:train_end]
            X_va, y_va = X_train_i[-val_size:], y_train_i[-val_size:]
            y_tr_spread = y_train_spread_i[:train_end]
            y_va_spread = y_train_spread_i[-val_size:]
        else:
            X_tr, y_tr = X_train_i, y_train_i
            X_va, y_va = X_test_i, y_test_i
            y_tr_spread = y_train_spread_i
            y_va_spread = y_test_spread_i

        print(f"📊 Thông tin tập dữ liệu cho {ticker}:")
        print(f"   🔹 Train : {len(X_tr)} mẫu")
        print(f"   🔹 Val   : {len(X_va)} mẫu")
        print(f"   🔹 Test  : {len(X_test_i)} mẫu")

        # ── Load hyperparams ──────────────────────────────────────────
        d_model      = 128
        num_heads    = 8
        dropout_rate = 0.3
        learning_rate = 1e-4
        batch_size   = 64

        for path in [
            os.path.join(ROOT_DIR, 'config', f'best_transformer_params_{ticker}.json'),
            os.path.join(ROOT_DIR, 'config', 'best_transformer_params.json'),
        ]:
            if os.path.exists(path):
                try:
                    import json
                    with open(path) as f:
                        p = json.load(f)
                    d_model       = p.get('d_model',       d_model)
                    num_heads     = p.get('num_heads',     num_heads)
                    dropout_rate  = p.get('dropout_rate',  dropout_rate)
                    learning_rate = p.get('learning_rate', learning_rate)
                    batch_size    = p.get('batch_size',    batch_size)
                    print(f"🥇 [TUNED] Dùng siêu tham số từ {os.path.basename(path)}:")
                    print(f"   d_model={d_model}, heads={num_heads}, dropout={dropout_rate:.4f}, "
                          f"lr={learning_rate:.6f}, batch={batch_size}")
                except Exception as e:
                    print(f"⚠️ Không đọc được {path}: {e}")
                break

        # ── Train Transformer ─────────────────────────────────────────
        print(f"🤖 [TRAIN] Đang huấn luyện Transformer cho {ticker}...")
        transformer_model = build_transformer(
            input_shape=(X_tr.shape[1], X_tr.shape[2]),
            d_model=d_model,
            heads=num_heads,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )

        history = transformer_model.fit(
            X_tr,
            {"output_return": y_tr, "output_spread": y_tr_spread},
            validation_data=(
                X_va,
                {"output_return": y_va, "output_spread": y_va_spread},
            ),
            epochs=120,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        # ── Evaluate ──────────────────────────────────────────────────
        split_idx     = int(len(X_3D) * 0.8)
        df_align      = df.iloc[dt.time_steps:].reset_index(drop=True)
        test_close_i  = df_align.loc[split_idx:, 'close'].values[:len(X_test_i)]
        y_test_true_i = test_close_i * (1 + y_test_raw_i)

        trans_preds_output = transformer_model.predict(X_test_i, verbose=0)
        trans_scaled = trans_preds_output[0] if isinstance(trans_preds_output, list) else trans_preds_output
        trans_return = dt.target_scaler.inverse_transform(trans_scaled).ravel()
        trans_preds  = test_close_i * (1 + trans_return)

        print(f"\n📊 BẢNG ĐÁNH GIÁ SAI SỐ CHO MÃ: {ticker}")
        evaluate_predictions(y_test_true_i, trans_preds, f"Transformer ({ticker})", ticker)

        # ── Biểu đồ kết quả ──────────────────────────────────────────
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_true_i[-100:], label="Giá thực tế (VNĐ)", color='black', linewidth=2.5)
        plt.plot(trans_preds[-100:],   label="Dự báo Transformer (VNĐ)", color='blue', linestyle='-.')
        plt.ylabel("Giá (VNĐ)", fontsize=12)
        plt.title(f"GIÁ THỰC TẾ VS DỰ BÁO — {ticker} (TRANSFORMER)", fontsize=14, fontweight='bold')
        plt.xlabel("100 Phiên giao dịch cuối (Tập Test)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(results_dir, f'transformer_result_{ticker}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 Biểu đồ kết quả: {plot_path}")

        # ── Biểu đồ loss ─────────────────────────────────────────────
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'],     label='Loss Huấn luyện')
        plt.plot(history.history['val_loss'], label='Loss Kiểm chứng')
        plt.title(f"LOSS CURVE — {ticker}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Huber)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        loss_path = os.path.join(results_dir, f'transformer_loss_{ticker}.png')
        plt.savefig(loss_path, dpi=300)
        plt.close()
        print(f"💾 Biểu đồ loss: {loss_path}")

        # ── Lưu model ────────────────────────────────────────────────
        # SỬA: dùng save_multitask_model thay vì transformer_model.save()
        # → lưu backbone Functional (serialize an toàn, load được)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        path_ts     = os.path.join(models_dir, f'transformer_model_{ticker}_{timestamp}.keras')
        path_latest = os.path.join(models_dir, f'transformer_model_{ticker}.keras')

        transformer_model.save(path_ts)
        transformer_model.save(path_latest)

        joblib.dump(dt.feature_scaler,
                    os.path.join(models_dir, f'feature_scaler_{ticker}_{timestamp}.pkl'))
        joblib.dump(dt.feature_scaler,
                    os.path.join(models_dir, f'feature_scaler_{ticker}.pkl'))
        joblib.dump(dt.target_scaler,
                    os.path.join(models_dir, f'target_scaler_{ticker}_{timestamp}.pkl'))
        joblib.dump(dt.target_scaler,
                    os.path.join(models_dir, f'target_scaler_{ticker}.pkl'))

        print(f"💾 Lưu xong model Transformer & Scalers cho {ticker} (timestamp: {timestamp})")
        print(f"   → {path_latest}")


if __name__ == "__main__":
    main()

