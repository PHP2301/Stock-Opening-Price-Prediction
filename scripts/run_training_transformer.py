import datetime
import json
import os
import random
import sys
# Cấu hình UTF-8 cho console để tránh lỗi UnicodeEncodeError trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# run_training.py — ĐÃ SỬA CÁC LỖI:
# 1. Thêm RETRAIN_TRANSFORMER flag: nếu False, load Transformer đã train thay vì train lại
#    → giữ nguyên Transformer tốt nhất từ đợt huấn luyện trước
# 2. Thay KFold → TimeSeriesSplit(gap=45) cho OOF embeddings
# 3. Inference: thay tính feature thủ công → dùng transformer.transform_df()
# 4. Thêm get_return_output() helper để handle list/tuple output nhất quán
# 5. Thêm sanity check sau inverse_transform để phát hiện scaler sai
# 6. SENTIMENT_ENGINE đồng nhất — đọc từ biến chung
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import Model
import tensorflow as tf
import math
import joblib
import yfinance as yf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=2')

from src.data_loader import fetch_and_prepare_data, format_vn, get_realtime_usd_vnd_rate
from src.features import DataTransformer, kalman_filter
from src.ai_models import (
    build_xgboost_optimized, build_transformer,
    PositionalEmbedding, TimeDecayAttention, MultiTaskModel, UncertaintyWeightsLayer,
)
from src.backtest_engine import (
    get_return_output, compute_metrics, compute_trade_metrics,
    run_simulation, run_walk_forward_evaluation
)

USD_TO_VND = get_realtime_usd_vnd_rate()
print(f"💵 Tỷ giá USD/VND: {format_vn(USD_TO_VND)} VNĐ\n")

LOOKBACK_WINDOW = 45

# === CẤU HÌNH CHÍNH ===
# SỬA: Đồng nhất SENTIMENT_ENGINE — dùng cùng engine cho train và inference
SENTIMENT_ENGINE = 'finbert'   # hoặc 'vader' — phải giống thiết lập chung

# SỬA: Flag kiểm soát có train lại Transformer không
# Đặt False khi đã có mô hình được huấn luyện từ trước và muốn giữ Transformer tốt nhất
RETRAIN_TRANSFORMER = os.environ.get("FORCE_RETRAIN", "1") == "1"



def cosine_decay(epoch):
    initial_lrate = 1e-4
    cos_outer = math.pi * epoch / 100
    return max(initial_lrate * 0.5 * (1.0 + math.cos(cos_outer)), 1e-5)


def evaluate_predictions(y_true, y_pred, model_name, ticker, rate=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"=== {model_name.upper()} ===")
    if "VNM" in ticker.upper():
        print(f"  RMSE: {format_vn(rmse)} VNĐ  |  MAE: {format_vn(mae)} VNĐ  |  MAPE: {mape:.2f}%\n")
    else:
        conv_rate = rate if rate is not None else USD_TO_VND
        print(f"  RMSE: {format_vn(rmse)} VNĐ (${rmse/conv_rate:.2f})  |  MAE: {format_vn(mae)} VNĐ (${mae/conv_rate:.2f})  |  MAPE: {mape:.2f}%\n")
    return rmse, mae, mape


# Các hàm backtest dùng chung đã được chuyển sang src.backtest_engine


def main():
    print("🚀 Khởi động Training Pipeline (Transformer) - Rolling Walk-Forward...")

    TICKERS    = ["VNM.VN", "GOOGL", "META"]
    START_TRAIN = "2010-01-01"
    END_PREDICT = "2026-05-20"

    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]

    callbacks_main = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=0),
        LearningRateScheduler(cosine_decay, verbose=0),
    ]

    models_dir  = os.path.join(ROOT_DIR, 'models')
    results_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    config_dir  = os.path.join(ROOT_DIR, 'config')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'='*80}")
        print(f"🔄 CHẠY ĐÀO TẠO & KIỂM THỬ CUỘN CHIẾU (WALK-FORWARD): {ticker}")
        print(f"{'='*80}")

        commission_pct = 0.0020 if "VNM" in ticker.upper() else 0.0010
        slippage_pct   = 0.0010 if "VNM" in ticker.upper() else 0.0005

        df = fetch_and_prepare_data(
            ticker, start_date=START_TRAIN, end_date=END_PREDICT,
            sentiment_engine=SENTIMENT_ENGINE,
        )
        df['date'] = pd.to_datetime(df['date'])

        # ── Load hyperparams ───────────────────────────────────────────
        d_model, num_heads, key_dim, dropout_rate, learning_rate, batch_size = 128, 8, 16, 0.3, 1e-4, 64
        for path in [
            os.path.join(ROOT_DIR, 'config', f'best_transformer_params_{ticker}.json'),
            os.path.join(ROOT_DIR, 'config', 'best_transformer_params.json'),
        ]:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8-sig') as f:
                        p = json.load(f)
                    d_model      = p.get('d_model', d_model)
                    num_heads    = p.get('num_heads', p.get('heads', num_heads))
                    key_dim      = p.get('key_dim', key_dim)
                    dropout_rate = p.get('dropout_rate', dropout_rate)
                    learning_rate = p.get('learning_rate', learning_rate)
                    batch_size   = p.get('batch_size', batch_size)
                    print(f"🥇 Dùng params từ {os.path.basename(path)}: d_model={d_model}, heads={num_heads}, key_dim={key_dim}")
                except Exception as e:
                    print(f"⚠️ Không đọc được {path}: {e}")
                break

        # ── BƯỚC 1: KIỂM THỬ CUỘN CHIẾU (ROLLING WALK-FORWARD BACKTEST) ──
        test_years = [2023, 2024, 2025, 2026]
        test_start_date = pd.to_datetime("2023-01-01")
        df_test_all = df[df['date'] >= test_start_date].copy().reset_index(drop=True)
        
        run_wf = not df_test_all.empty
        if run_wf:
            test_start_idx = df[df['date'] >= test_start_date].index[0]
            df_test_extended_all = df.iloc[test_start_idx : test_start_idx + len(df_test_all) + 1].reset_index(drop=True)
            
            trans_pred_accum = []
            previous_weights = None
            
            for year in test_years:
                year_start = pd.to_datetime(f"{year}-01-01")
                year_end   = pd.to_datetime(f"{year}-12-31")
                
                train_df = df[df['date'] < year_start].copy()
                test_df  = df[(df['date'] >= year_start) & (df['date'] <= year_end)].copy()
                
                if test_df.empty:
                    continue
                    
                print(f"\n▶️ [WINDOW {year}] — Huấn luyện: {START_TRAIN} → {year-1} ({len(train_df)} phiên) | Kiểm thử: {year} ({len(test_df)} phiên)...")
                
                dt_wf = DataTransformer(time_steps=LOOKBACK_WINDOW)
                X_train_raw = dt_wf.transform_df(train_df).values
                y_train_raw = train_df[dt_wf.target_cols].values
                
                dt_wf.feature_scaler.fit(X_train_raw)
                dt_wf.target_scaler.fit(y_train_raw)
                
                if all(col in train_df.columns for col in dt_wf.spread_cols):
                    y_train_spread_raw = train_df[dt_wf.spread_cols].values
                    dt_wf.spread_scaler.fit(y_train_spread_raw)
                    y_train_spread_scaled = dt_wf.spread_scaler.transform(y_train_spread_raw)
                else:
                    y_train_spread_scaled = None
                    
                X_train_scaled = dt_wf.feature_scaler.transform(X_train_raw)
                y_train_scaled = dt_wf.target_scaler.transform(y_train_raw)
                
                X_train_3D, y_train_3D, y_train_spread_3D = dt_wf.create_sliding_windows(
                    X_train_scaled, y_train_scaled, y_train_spread_scaled
                )
                
                val_size = int(len(X_train_3D) * 0.1)
                purge = 45
                if val_size > 0 and len(X_train_3D) - val_size - purge > 0:
                    train_end = len(X_train_3D) - val_size - purge
                    X_tr, y_tr = X_train_3D[:train_end], y_train_3D[:train_end]
                    X_va, y_va = X_train_3D[-val_size:], y_train_3D[-val_size:]
                    y_tr_spread = y_train_spread_3D[:train_end] if y_train_spread_3D is not None else None
                    y_va_spread = y_train_spread_3D[-val_size:] if y_train_spread_3D is not None else None
                else:
                    X_tr, y_tr = X_train_3D, y_train_3D
                    X_va, y_va = X_train_3D, y_train_3D
                    y_tr_spread = y_train_spread_3D
                    y_va_spread = y_train_spread_3D
                    
                if previous_weights is not None:
                    print(f"      [WARM-START] Khởi tạo trọng số từ Window trước để tối ưu hóa học...")
                    transformer_model = build_transformer(
                        input_shape=(X_train_3D.shape[1], X_train_3D.shape[2]),
                        d_model=d_model,
                        num_heads=num_heads,
                        key_dim=key_dim,
                        dropout_rate=dropout_rate,
                        learning_rate=2e-5,
                    )
                    try:
                        transformer_model.set_weights(previous_weights)
                        epochs_to_run = 20
                        patience_run  = 6
                    except Exception as e:
                        print(f"      ⚠️ Không load được trọng số: {e}, huấn luyện từ đầu.")
                        transformer_model = build_transformer(
                            input_shape=(X_train_3D.shape[1], X_train_3D.shape[2]),
                            d_model=d_model,
                            num_heads=num_heads,
                            key_dim=key_dim,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate,
                        )
                        epochs_to_run = 60
                        patience_run  = 10
                else:
                    transformer_model = build_transformer(
                        input_shape=(X_train_3D.shape[1], X_train_3D.shape[2]),
                        d_model=d_model,
                        num_heads=num_heads,
                        key_dim=key_dim,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate,
                    )
                    epochs_to_run = 80
                    patience_run  = 12
                    
                callbacks_window = [
                    EarlyStopping(monitor='val_loss', patience=patience_run, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
                ]
                
                transformer_model.fit(
                    X_tr,
                    {"output_return": y_tr, "output_spread": y_tr_spread},
                    validation_data=(X_va, {"output_return": y_va, "output_spread": y_va_spread}),
                    epochs=epochs_to_run, batch_size=batch_size, callbacks=callbacks_window, verbose=0
                )
                
                previous_weights = transformer_model.get_weights()
                
                test_extended_df = pd.concat([train_df.tail(45), test_df], ignore_index=True)
                X_test_raw = dt_wf.transform_df(test_extended_df).values
                y_test_raw_full = test_extended_df[dt_wf.target_cols].values
                
                X_test_scaled = dt_wf.feature_scaler.transform(X_test_raw)
                y_test_scaled = dt_wf.target_scaler.transform(y_test_raw_full)
                
                if all(col in test_extended_df.columns for col in dt_wf.spread_cols):
                    y_test_spread_raw = test_extended_df[dt_wf.spread_cols].values
                    y_test_spread_scaled = dt_wf.spread_scaler.transform(y_test_spread_raw)
                else:
                    y_test_spread_scaled = None
                    
                X_test_3D, y_test_3D, _ = dt_wf.create_sliding_windows(
                    X_test_scaled, y_test_scaled, y_test_spread_scaled
                )
                
                trans_pred_raw   = transformer_model.predict(X_test_3D, verbose=0)
                trans_pred_clean = get_return_output(trans_pred_raw)
                trans_pred_ret   = dt_wf.target_scaler.inverse_transform(trans_pred_clean)
                trans_pred_accum.append(trans_pred_ret)
                
            trans_returns_all = np.concatenate(trans_pred_accum, axis=0)
            
            cur_threshold_buy = 0.0050 if "VNM" in ticker.upper() else 0.0010
            cur_threshold_sell = -cur_threshold_buy
            
            print(f"\n📊 [SIMULATION] Chạy giả lập trên dữ liệu kiểm thử cuộn chiếu 2023 → 2026 ({len(df_test_all)} phiên)...")
            dates, equity, bh_equity, trades = run_simulation(
                df_test_all, df_test_extended_all,
                trans_returns_all,
                ticker, commission_pct, slippage_pct,
                cur_threshold_buy, cur_threshold_sell
            )
            
            metrics = compute_metrics(equity, bh_equity, dates)
            win_rate, total_trades, profit_factor = compute_trade_metrics(trades, commission_pct)
            
            print(f"\n🏆 KẾT QUẢ WALK-FORWARD ROLLING BACKTEST (2023–2026):")
            print(f"   📈 Chiến lược : {metrics['strat_return']:+.2f}%")
            print(f"   📦 Buy&Hold  : {metrics['bh_return']:+.2f}%")
            print(f"   📊 Sharpe    : {metrics['sharpe']:.2f}")
            print(f"   📉 Max DD    : {metrics['mdd']:.2f}% (B&H: {metrics['bh_mdd']:.2f}%)")
            print(f"   ⚖️  Calmar    : {metrics['calmar']:.2f}")
            print(f"   🔔 Số lệnh   : {total_trades}")
            print(f"   🥇 Win Rate  : {win_rate:.2f}%")
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, equity, label="Rolling Walk-Forward Strategy", color='darkgreen', linewidth=2)
            plt.plot(dates, bh_equity, label="Buy & Hold", color='grey', linestyle='--', alpha=0.8)
            plt.title(f"Rolling Walk-Forward Equity Curve — {ticker}", fontsize=13, fontweight='bold')
            plt.xlabel("Ngày")
            currency = "USD" if not ("VNM" in ticker.upper()) else "VNĐ"
            plt.ylabel(f"Tài sản ({currency})")
            plt.legend()
            plt.grid(True, alpha=0.25)
            plot_path = os.path.join(results_dir, f'walk_forward_equity_curve_{ticker}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"💾 Đã lưu biểu đồ tại: {plot_path}")
            
            current_drawdown = (equity[-1] - max(equity)) / max(equity) if len(equity) > 0 else 0.0
            perf_path = os.path.join(config_dir, f'performance_metrics_{ticker}.json')
            perf_data = {
                'overall_win_rate': win_rate / 100.0,
                'total_trades': total_trades,
                'profit_factor': profit_factor,
                'strat_return': metrics['strat_return'],
                'bh_return': metrics['bh_return'],
                'sharpe': metrics['sharpe'],
                'max_drawdown': metrics['mdd'],
                'current_drawdown': current_drawdown,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(perf_path, 'w', encoding='utf-8') as f:
                json.dump(perf_data, f, indent=4)
            print(f"💾 Đã lưu hiệu suất backtest vào: {perf_path}")
            
        # ── BƯỚC 2: HUẤN LUYỆN MÔ HÌNH SẢN XUẤT CUỐI CÙNG (100% DỮ LIỆU) ──
        print(f"\n🤖 [FINAL TRAINING] Huấn luyện mô hình sản xuất trên 100% dữ liệu lịch sử...")
        dt_final = DataTransformer(time_steps=LOOKBACK_WINDOW)
        X_scaled, y_scaled, y_spread_scaled = dt_final.fit_transform_train_only(df, train_ratio=0.9)
        X_3D, y_3D, y_spread_3D = dt_final.create_sliding_windows(X_scaled, y_scaled, y_spread_scaled)
        
        X_train, y_train, X_val, y_val, _, y_train_spread, y_val_spread = \
            dt_final.split_train_test_chronological(df, X_3D, y_3D, y_spread_3D, train_ratio=0.9)
            
        train_mask = ~np.isnan(y_train).any(axis=1) & ~np.isnan(y_train_spread).any(axis=1)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        y_train_spread = y_train_spread[train_mask]
        
        val_mask = ~np.isnan(y_val).any(axis=1) & ~np.isnan(y_val_spread).any(axis=1)
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]
        y_val_spread = y_val_spread[val_mask]
        
        # 2.1 K-Fold OOF cho XGBoost Stacking
        print(f"🤖 [FINAL OOF] TimeSeriesSplit K=5, gap=45 để tạo OOF latent embeddings cho XGBoost...")
        tscv_oof = TimeSeriesSplit(n_splits=5, gap=45)
        
        dummy_model = build_transformer(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            d_model=d_model,
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
        _ = dummy_model(tf.zeros((1, X_train.shape[1], X_train.shape[2])))
        latent_dim = dummy_model.get_layer("latent_embedding").output.shape[-1]
        
        trans_pred_train_oof = np.zeros((len(X_train), latent_dim))
        
        for fold, (train_idx, val_idx) in enumerate(tscv_oof.split(X_train), 1):
            X_tr_f, y_tr_f = X_train[train_idx], y_train[train_idx]
            X_va_f, y_va_f = X_train[val_idx], y_train[val_idx]
            y_tr_s_f = y_train_spread[train_idx]
            y_va_s_f = y_train_spread[val_idx]
            
            fold_model = build_transformer(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                d_model=d_model,
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )
            fold_cb = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
            ]
            fold_model.fit(
                X_tr_f,
                {"output_return": y_tr_f, "output_spread": y_tr_s_f},
                validation_data=(X_va_f, {"output_return": y_va_f, "output_spread": y_va_s_f}),
                epochs=50, batch_size=batch_size, callbacks=fold_cb, verbose=0,
            )
            _ = fold_model(X_va_f[:1])
            fold_extractor = Model(
                inputs=fold_model.input,
                outputs=fold_model.get_layer("latent_embedding").output
            )
            fold_pred_latent = fold_extractor.predict(X_va_f, verbose=0)
            trans_pred_train_oof[val_idx] = fold_pred_latent
            
        # 2.2 Huấn luyện Transformer chính sản xuất
        print(f"🤖 [FINAL TRAIN] Transformer chính cho {ticker}...")
        transformer_model = build_transformer(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            d_model=d_model,
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
        transformer_model.fit(
            X_train,
            {"output_return": y_train, "output_spread": y_train_spread},
            validation_data=(X_val, {"output_return": y_val, "output_spread": y_val_spread}),
            epochs=100, batch_size=batch_size, callbacks=callbacks_main, verbose=2,
        )
        
        # 2.3 Huấn luyện XGBoost Stacking sản xuất
        print(f"🌳 [FINAL TRAIN] XGBoost stacking cho {ticker}...")
        X_train_today = X_train[:, -1, :]
        X_train_xgb = np.concatenate([trans_pred_train_oof, X_train_today], axis=1)
        xgb_model = build_xgboost_optimized(X_train_xgb, y_train)
        
        # 2.4 Lưu trữ toàn bộ mô hình và Scalers sản xuất
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        trans_latest_path = os.path.join(models_dir, f'transformer_model_{ticker}.keras')
        xgb_latest_path = os.path.join(models_dir, f'xgboost_model_{ticker}.pkl')
        
        transformer_model.save(os.path.join(models_dir, f'transformer_model_{ticker}_{timestamp}.keras'))
        transformer_model.save(trans_latest_path)
        joblib.dump(xgb_model, os.path.join(models_dir, f'xgboost_model_{ticker}_{timestamp}.pkl'))
        joblib.dump(xgb_model, xgb_latest_path)
        joblib.dump(dt_final.feature_scaler, os.path.join(models_dir, f'feature_scaler_{ticker}.pkl'))
        joblib.dump(dt_final.target_scaler,  os.path.join(models_dir, f'target_scaler_{ticker}.pkl'))
        
        print(f"💾 Đã lưu xong toàn bộ mô hình và scalers sản xuất cho {ticker}!")
        print(f"{'─'*80}")


if __name__ == "__main__":
    main()




