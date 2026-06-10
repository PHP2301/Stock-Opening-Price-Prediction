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
#    → giữ nguyên Transformer tốt nhất từ run_training_transformer.py
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

USD_TO_VND = get_realtime_usd_vnd_rate()
print(f"💵 Tỷ giá USD/VND: {format_vn(USD_TO_VND)} VNĐ\n")

LOOKBACK_WINDOW = 45

# === CẤU HÌNH CHÍNH ===
# SỬA: Đồng nhất SENTIMENT_ENGINE — dùng cùng engine cho train và inference
SENTIMENT_ENGINE = 'finbert'   # hoặc 'vader' — phải giống run_training_transformer.py

# SỬA: Flag kiểm soát có train lại Transformer không
# Đặt False khi đã chạy run_training_transformer.py và muốn giữ Transformer tốt nhất
RETRAIN_TRANSFORMER = os.environ.get("FORCE_RETRAIN", "1") == "1"


def get_return_output(pred):
    """
    SỬA: Helper chuẩn hóa output của Transformer.
    Transformer có thể trả list [output_return, output_spread] hoặc tensor đơn.
    Luôn lấy output_return (index 0).
    """
    return pred[0] if isinstance(pred, (list, tuple)) else pred


def cosine_decay(epoch):
    initial_lrate = 1e-4
    cos_outer = math.pi * epoch / 100
    return max(initial_lrate * 0.5 * (1.0 + math.cos(cos_outer)), 1e-5)


def evaluate_predictions(y_true, y_pred, model_name, ticker):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"=== {model_name.upper()} ===")
    if "VNM" in ticker.upper():
        print(f"  RMSE: {format_vn(rmse)} VNĐ  |  MAE: {format_vn(mae)} VNĐ  |  MAPE: {mape:.2f}%\n")
    else:
        print(f"  RMSE: {format_vn(rmse)} VNĐ (${rmse/USD_TO_VND:.2f})  |  MAE: {format_vn(mae)} VNĐ  |  MAPE: {mape:.2f}%\n")
    return rmse, mae, mape


def log_prediction_to_file(ticker, last_date, last_close, risk_level, risk_ratio,
                            xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper,
                            rate_today=None):
    logs_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path  = os.path.join(logs_dir, "predictions_history.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def trend(pred_val, ref):
        pct   = (pred_val - ref) / ref * 100
        emoji = "📈 TĂNG" if pred_val >= ref else "📉 GIẢM"
        return f"({emoji} {pct:+.2f}%)"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== DỰ BÁO ({timestamp}) ===\n")
        f.write(f"Mã: {ticker}\n")
        rate = rate_today or 25400
        f.write(f"Đóng cửa ({last_date}): {format_vn(last_close)} VNĐ\n")
        f.write(f"Rủi ro: {risk_level} ({risk_ratio:.2f}%)\n")
        f.write(f"XGBoost : {format_vn(xgb_val)} VNĐ {trend(xgb_val, last_close)} | [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}]\n")
        f.write(f"Transformer: {format_vn(trans_val)} VNĐ {trend(trans_val, last_close)} | [{format_vn(trans_lower)} - {format_vn(trans_upper)}]\n")
        f.write("-" * 50 + "\n\n")


def main():
    print("🚀 Khởi động Hybrid Training Pipeline (Transformer → XGBoost)...")

    TICKERS    = ["VNM.VN", "GOOGL", "META"]
    START_TRAIN = "2010-01-01"
    END_PREDICT = "2026-05-20"

    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]

    callbacks_main = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
        LearningRateScheduler(cosine_decay, verbose=0),
    ]

    models_dir  = os.path.join(ROOT_DIR, 'models')
    results_dir = os.path.join(ROOT_DIR, 'reports', 'figures')

    for ticker in TICKERS:
        print(f"\n{'='*55}")
        print(f"🔄 {ticker}")
        print(f"{'='*55}")

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

        # ── Load hyperparams ───────────────────────────────────────────
        d_model, num_heads, dropout_rate, learning_rate, batch_size = 128, 8, 0.3, 1e-4, 64
        for path in [
            os.path.join(ROOT_DIR, 'config', f'best_transformer_params_{ticker}.json'),
            os.path.join(ROOT_DIR, 'config', 'best_transformer_params.json'),
        ]:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        p = json.load(f)
                    d_model      = p.get('d_model', d_model)
                    num_heads    = p.get('num_heads', p.get('heads', num_heads))
                    dropout_rate = p.get('dropout_rate', dropout_rate)
                    learning_rate = p.get('learning_rate', learning_rate)
                    batch_size   = p.get('batch_size', batch_size)
                    print(f"🥇 Dùng params từ {os.path.basename(path)}: d_model={d_model}, heads={num_heads}")
                except Exception as e:
                    print(f"⚠️ Không đọc được {path}: {e}")
                break

        trans_latest_path = os.path.join(models_dir, f'transformer_model_{ticker}.keras')

        # ════════════════════════════════════════════════════════════════
        # SỬA: RETRAIN_TRANSFORMER flag
        # Nếu False + file đã tồn tại → load Transformer từ run_training_transformer.py
        # Không ghi đè kết quả tốt nhất
        # ════════════════════════════════════════════════════════════════
        if not RETRAIN_TRANSFORMER and os.path.exists(trans_latest_path):
            print(f"♻️  [LOAD] Dùng Transformer đã train: {trans_latest_path}")
            from src.ai_models import PositionalEmbedding, TimeDecayAttention, UncertaintyWeightsLayer, MultiTaskModel
            custom_objects = {
                'PositionalEmbedding': PositionalEmbedding,
                'TimeDecayAttention': TimeDecayAttention,
                'UncertaintyWeightsLayer': UncertaintyWeightsLayer,
                'MultiTaskModel': MultiTaskModel,
            }
            transformer_model = tf.keras.models.load_model(
                trans_latest_path,
                custom_objects=custom_objects,
                safe_mode=False
            )
            # Load scalers đã lưu (nếu có) để đồng bộ
            feat_scaler_path = os.path.join(models_dir, f'feature_scaler_{ticker}.pkl')
            targ_scaler_path = os.path.join(models_dir, f'target_scaler_{ticker}.pkl')
            if os.path.exists(feat_scaler_path):
                dt.feature_scaler = joblib.load(feat_scaler_path)
            if os.path.exists(targ_scaler_path):
                dt.target_scaler = joblib.load(targ_scaler_path)
        else:
            # ── BƯỚC 1: K-Fold OOF cho XGBoost ─────────────────────────
            # SỬA: TimeSeriesSplit(gap=45) thay vì KFold(shuffle=False)
            # Đảm bảo thứ tự thời gian và có purge gap
            print(f"🤖 [OOF] TimeSeriesSplit K=5, gap=45 để tạo OOF embeddings...")
            tscv = TimeSeriesSplit(n_splits=5, gap=45)
            X_train_latent_oof = np.zeros((len(X_train_i), 32))

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_i), 1):
                print(f"   Fold {fold}/5 — train={len(train_idx)}, val={len(val_idx)}")
                X_tr_f = X_train_i[train_idx];  y_tr_f = y_train_i[train_idx]
                X_va_f = X_train_i[val_idx];    y_va_f = y_train_i[val_idx]
                y_tr_s_f = y_train_spread_i[train_idx]
                y_va_s_f = y_train_spread_i[val_idx]

                fold_model = build_transformer(
                    input_shape=(X_train_i.shape[1], X_train_i.shape[2]),
                    d_model=d_model, dropout_rate=dropout_rate, learning_rate=learning_rate,
                )
                fold_cb = [
                    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
                ]
                fold_model.fit(
                    X_tr_f,
                    {"output_return": y_tr_f, "output_spread": y_tr_s_f},
                    validation_data=(X_va_f, {"output_return": y_va_f, "output_spread": y_va_s_f}),
                    # SỬA: tăng epochs fold từ 35 → 60 để fold embedding gần với main model hơn
                    epochs=60, batch_size=batch_size, callbacks=fold_cb, verbose=0,
                )
                fold_extractor = Model(
                    inputs=fold_model.inputs,
                    outputs=fold_model.get_layer("latent_embedding").output,
                )
                X_train_latent_oof[val_idx] = fold_extractor.predict(X_va_f, verbose=0)

            # ── BƯỚC 2: Train Transformer chính ─────────────────────────
            print(f"🤖 [TRAIN] Transformer chính cho {ticker}...")
            transformer_model = build_transformer(
                input_shape=(X_tr.shape[1], X_tr.shape[2]),
                d_model=d_model, dropout_rate=dropout_rate, learning_rate=learning_rate,
            )
            transformer_model.fit(
                X_tr,
                {"output_return": y_tr, "output_spread": y_tr_spread},
                validation_data=(X_va, {"output_return": y_va, "output_spread": y_va_spread}),
                epochs=100, batch_size=batch_size, callbacks=callbacks_main, verbose=0,
            )
        # ── BƯỚC 3: Feature extractor & XGBoost ───────────────────────
        print(f"🔍 [EXTRACT] Trích latent features...")
        feature_extractor = Model(
            inputs=transformer_model.inputs,   # ← đổi .input → .inputs
            outputs=transformer_model.get_layer("latent_embedding").output,
        )

        if RETRAIN_TRANSFORMER or not os.path.exists(trans_latest_path):
            X_train_today_all  = X_train_i[:, -1, :]
            X_train_hybrid_all = np.concatenate([X_train_latent_oof, X_train_today_all], axis=1)
        else:
            # Load mode: tính OOF từ Transformer đã load (không có X_train_latent_oof)
            X_train_latent_all = feature_extractor.predict(X_train_i, verbose=0)
            X_train_today_all  = X_train_i[:, -1, :]
            X_train_hybrid_all = np.concatenate([X_train_latent_all, X_train_today_all], axis=1)
            print("  [INFO] Load mode: dùng main Transformer latent (không OOF)")

        print(f"🌳 [TRAIN] XGBoost hybrid ({X_train_hybrid_all.shape[1]}-dim)...")
        xgb_model = build_xgboost_optimized(X_train_hybrid_all, y_train_i)

        # ── BƯỚC 4: Evaluate ──────────────────────────────────────────
        split_idx     = int(len(X_3D) * 0.8)
        df_align      = df.iloc[dt.time_steps:].reset_index(drop=True)
        test_start    = split_idx + 45
        test_close_i  = df_align.loc[test_start:, 'close'].values[:len(X_test_i)]
        y_test_true_i = test_close_i * (1 + y_test_raw_i)

        X_test_latent  = feature_extractor.predict(X_test_i, verbose=0)
        X_test_today   = X_test_i[:, -1, :]
        X_test_hybrid  = np.concatenate([X_test_latent, X_test_today], axis=1)

        # SỬA: dùng get_return_output() để handle list/tensor nhất quán
        hybrid_xgb_scaled = xgb_model.predict(X_test_hybrid).reshape(-1, 1)
        hybrid_xgb_return = dt.target_scaler.inverse_transform(hybrid_xgb_scaled).ravel()

        # SỬA: sanity check — return > 20%/ngày là scaler sai
        assert np.abs(hybrid_xgb_return).max() < 0.20, \
            f"[SANITY] XGBoost return bất thường: {np.abs(hybrid_xgb_return).max():.4f} — kiểm tra target_scaler!"

        hybrid_xgb_preds = test_close_i * (1 + hybrid_xgb_return)

        trans_pred_raw   = transformer_model.predict(X_test_i, verbose=0)
        trans_pred_clean = get_return_output(trans_pred_raw)
        trans_return     = dt.target_scaler.inverse_transform(trans_pred_clean).ravel()
        trans_preds      = test_close_i * (1 + trans_return)

        print(f"\n📊 KẾT QUẢ — {ticker}")
        evaluate_predictions(y_test_true_i, hybrid_xgb_preds, f"Hybrid XGBoost ({ticker})", ticker)
        evaluate_predictions(y_test_true_i, trans_preds,      f"Transformer ({ticker})", ticker)

        # ── Biểu đồ ──────────────────────────────────────────────────
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_true_i[-100:], label="Giá thực tế", color='black', linewidth=2.5)
        plt.plot(hybrid_xgb_preds[-100:], label="Hybrid XGBoost", color='red', linestyle='--')
        plt.plot(trans_preds[-100:],      label="Transformer",    color='blue', linestyle='-.')
        plt.title(f"{ticker} — 100 phiên cuối tập Test", fontsize=14, fontweight='bold')
        plt.xlabel("Phiên"); plt.ylabel("Giá (VNĐ)")
        plt.legend(); plt.grid(True, alpha=0.3)
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f'model_battle_result_{ticker}.png'), dpi=300)
        plt.close()

        # ── Lưu mô hình ──────────────────────────────────────────────
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        joblib.dump(xgb_model, os.path.join(models_dir, f'xgboost_model_{ticker}_{timestamp}.pkl'))
        joblib.dump(xgb_model, os.path.join(models_dir, f'xgboost_model_{ticker}.pkl'))

        # Chỉ lưu Transformer nếu đã train lại
        if RETRAIN_TRANSFORMER or not os.path.exists(trans_latest_path):
            transformer_model.save(os.path.join(models_dir, f'transformer_model_{ticker}_{timestamp}.keras'))
            transformer_model.save(trans_latest_path)
            joblib.dump(dt.feature_scaler, os.path.join(models_dir, f'feature_scaler_{ticker}.pkl'))
            joblib.dump(dt.target_scaler,  os.path.join(models_dir, f'target_scaler_{ticker}.pkl'))

        print(f"💾 Lưu xong mô hình cho {ticker} (timestamp: {timestamp})")

        # ── Live prediction ──────────────────────────────────────────
        print(f"\n🔮 Dự báo giá đóng cửa 3 ngày tới (T+3) — {ticker}...")
        raw_df = yf.download(ticker, period="150d", progress=False)
        if raw_df.empty:
            print(f"  Không tải được dữ liệu live cho {ticker}")
            continue

        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.droplevel(1)
        raw_df.columns = [c.lower() for c in raw_df.columns]

        usd_vnd_rate = USD_TO_VND
        # Quy đổi USD→VND cho non-VNM
        if "VNM" not in ticker.upper():
            try:
                df_rate_hist = yf.download("USDVND=X", period="150d", progress=False)
                if not df_rate_hist.empty:
                    if isinstance(df_rate_hist.columns, pd.MultiIndex):
                        df_rate_hist.columns = [c[0].lower() for c in df_rate_hist.columns]
                    else:
                        df_rate_hist.columns = [c.lower() for c in df_rate_hist.columns]
                    df_rate_hist = df_rate_hist[['close']].rename(columns={'close': 'rate_close'})
                    df_rate_hist['rate_close'] = df_rate_hist['rate_close'].apply(
                        lambda x: x * 1000.0 if x < 1000.0 else x
                    )
                    df_rate_hist.loc[
                        (df_rate_hist['rate_close'] < 15000) | (df_rate_hist['rate_close'] > 28000),
                        'rate_close'
                    ] = np.nan
                    df_rate_hist['rate_close'] = df_rate_hist['rate_close'].ffill().bfill().fillna(USD_TO_VND)
                    usd_vnd_rate = float(df_rate_hist['rate_close'].iloc[-1])
                    raw_df = raw_df.merge(df_rate_hist, left_index=True, right_index=True, how='left')
                    raw_df['rate_close'] = raw_df['rate_close'].ffill().bfill().fillna(USD_TO_VND)
                    for col in ['open', 'high', 'low', 'close']:
                        raw_df[col] = raw_df[col] * raw_df['rate_close']
                    raw_df = raw_df.drop(columns=['rate_close'])
            except Exception as e:
                print(f"  [WARNING] Không quy đổi tỷ giá được: {e}")
                for col in ['open', 'high', 'low', 'close']:
                    raw_df[col] = raw_df[col] * USD_TO_VND

        # Thêm macro vĩ mô
        raw_df = raw_df.reset_index()
        raw_df.columns = [c.lower() for c in raw_df.columns]
        raw_df['date'] = pd.to_datetime(raw_df['date'])

        start_live = raw_df['date'].min().strftime('%Y-%m-%d')
        end_live   = (raw_df['date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        def _quick_macro(sym, col, pct=False):
            try:
                r = yf.download(sym, start=start_live, end=end_live, progress=False)
                if r.empty: raise ValueError("empty")
                tmp = r.reset_index()
                if isinstance(tmp.columns, pd.MultiIndex):
                    tmp.columns = [str(c[0]).lower() for c in tmp.columns]
                else:
                    tmp.columns = [str(c).lower() for c in tmp.columns]
                tmp['date'] = pd.to_datetime(tmp['date'])
                tmp[col] = tmp['close'].pct_change() if pct else tmp['close']
                return tmp[['date', col]]
            except Exception:
                return pd.DataFrame(columns=['date', col])

        is_vn = ".VN" in ticker.upper()
        idx_sym = "VNM" if is_vn else "^GSPC"
        raw_df = raw_df.merge(_quick_macro(idx_sym, 'market_return', pct=True), on='date', how='left')
        raw_df = raw_df.merge(_quick_macro("^VIX",    'vix'),                   on='date', how='left')
        raw_df = raw_df.merge(_quick_macro("^TNX",    'bond_yield_10y'),        on='date', how='left')
        raw_df = raw_df.merge(_quick_macro("DX-Y.NYB",'dollar_index_change',pct=True), on='date', how='left')

        raw_df['market_return']    = raw_df['market_return'].fillna(0.0)
        raw_df['vix']              = raw_df['vix'].ffill().bfill().fillna(20.0)
        raw_df['bond_yield_10y']   = raw_df['bond_yield_10y'].ffill().bfill().fillna(4.0)
        raw_df['dollar_index_change'] = raw_df['dollar_index_change'].fillna(0.0)
        raw_df['usdvnd_change']    = 0.0

        if is_vn:
            raw_df['vix_lag1']            = raw_df['vix'].shift(1)
            raw_df['bond_yield_lag1']     = raw_df['bond_yield_10y'].shift(1)
            raw_df['vnindex_return_lag1'] = raw_df['market_return'].shift(1)
        else:
            raw_df['vix_lag1']            = raw_df['vix']
            raw_df['bond_yield_lag1']     = raw_df['bond_yield_10y']
            raw_df['vnindex_return_lag1'] = raw_df['market_return']

        raw_df['vix_lag1']        = raw_df['vix_lag1'].ffill().bfill().fillna(20.0)
        raw_df['bond_yield_lag1'] = raw_df['bond_yield_lag1'].ffill().bfill().fillna(4.0)
        raw_df['day_of_week_sin'] = np.sin(2 * np.pi * raw_df['date'].dt.dayofweek / 5)
        raw_df['day_of_week_cos'] = np.cos(2 * np.pi * raw_df['date'].dt.dayofweek / 5)
        raw_df['month_sin']       = np.sin(2 * np.pi * raw_df['date'].dt.month / 12)
        raw_df['month_cos']       = np.cos(2 * np.pi * raw_df['date'].dt.month / 12)
        raw_df['is_quarter_end']  = 0
        raw_df['days_before_tet'] = 30.0
        raw_df['sentiment_score'] = 0.0
        raw_df['news_volume']     = 0.0

        try:
            from src.news_sentiment import get_news_sentiment_features
            df_sent = get_news_sentiment_features(
                ticker, raw_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                engine=SENTIMENT_ENGINE,
            )
            df_sent['date'] = pd.to_datetime(df_sent['date'])
            raw_df = pd.merge(raw_df, df_sent, on='date', how='left')
            raw_df['sentiment_score'] = raw_df.get('sentiment_score_y', raw_df.get('sentiment_score', 0.0)).fillna(0.0)
            raw_df['news_volume']     = raw_df.get('news_volume_y', raw_df.get('news_volume', 0.0)).fillna(0.0)
        except Exception as e:
            print(f"  [WARNING] Tin tức: {e}")

        # === SỬA: dùng transform_df() thay vì tính feature thủ công ===
        # Trước: tính rsi_14, macd, bb... thủ công → shape mismatch với 34 features
        # Sau: gọi dt.transform_df() → đúng 34 features, nhất quán với training
        raw_df = raw_df.set_index('date')
        # close_smoothed được tự tính bên trong transform_df() nếu chưa có
        raw_df_reset = raw_df.reset_index()
        recent_features = dt.transform_df(raw_df_reset).tail(LOOKBACK_WINDOW)

        if len(recent_features) < LOOKBACK_WINDOW:
            print(f"  Không đủ {LOOKBACK_WINDOW} ngày dữ liệu live cho {ticker}")
            continue

        recent_scaled = dt.feature_scaler.transform(recent_features.values)
        X_predict     = recent_scaled.reshape(1, LOOKBACK_WINDOW, len(dt.feature_cols))

        # Dự báo hybrid
        X_pred_latent  = feature_extractor.predict(X_predict, verbose=0)
        X_pred_today   = X_predict[0, -1, :].reshape(1, -1)
        X_pred_hybrid  = np.concatenate([X_pred_latent, X_pred_today], axis=1)

        xgb_pred_scaled   = xgb_model.predict(X_pred_hybrid).reshape(-1, 1)
        xgb_return_future = dt.target_scaler.inverse_transform(xgb_pred_scaled)[0][0]

        trans_pred_live  = transformer_model.predict(X_predict, verbose=0)
        trans_pred_clean = get_return_output(trans_pred_live)
        trans_return_future = dt.target_scaler.inverse_transform(trans_pred_clean)[0][0]

        last_close = float(raw_df['close'].dropna().iloc[-1])
        last_date  = raw_df.index[-1].strftime('%Y-%m-%d')

        xgb_val   = last_close * (1 + xgb_return_future)
        trans_val = last_close * (1 + trans_return_future)

        last_atr   = raw_df_reset['high'].iloc[-14:].values - raw_df_reset['low'].iloc[-14:].values
        atr_approx = float(np.mean(np.abs(last_atr))) if len(last_atr) > 0 else last_close * 0.02
        risk_ratio = (atr_approx / last_close) * 100
        risk_level = (
            "Thấp 🟢" if risk_ratio < 1.5 else
            "Trung bình 🟡" if risk_ratio < 3.0 else
            "Cao 🔴 [CẢNH BÁO]"
        )

        xgb_lower, xgb_upper   = xgb_val - 1.5 * atr_approx, xgb_val + 1.5 * atr_approx
        trans_lower, trans_upper = trans_val - 1.5 * atr_approx, trans_val + 1.5 * atr_approx

        def trend_str(val):
            pct = (val - last_close) / last_close * 100
            return f"{'📈 TĂNG' if val >= last_close else '📉 GIẢM'} ({pct:+.2f}%)"

        print(f"  💵 Close ({last_date}): {format_vn(last_close)} VNĐ")
        print(f"  ⚠️  Rủi ro: {risk_level} ({risk_ratio:.2f}%)")
        print(f"  🌳 XGBoost   : {format_vn(xgb_val)} | {trend_str(xgb_val)}")
        print(f"  🤖 Transformer: {format_vn(trans_val)} | {trend_str(trans_val)}")

        log_prediction_to_file(
            ticker, last_date, last_close, risk_level, risk_ratio,
            xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper,
            usd_vnd_rate,
        )

        print(f"\n{'─'*55}")


if __name__ == "__main__":
    main()




