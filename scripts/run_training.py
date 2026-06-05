import datetime
import os
import random
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# Cấu hình encoding utf-8 cho console ngay từ đầu để tránh lỗi Unicode trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import Model
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '1')
# Enable XLA JIT for CPU training
import tensorflow as tf
import math
import joblib
import yfinance as yf
import pandas_ta as ta

# Cấu hình absolute root path để có thể chạy script từ bất kỳ đâu
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# ==========================================
# CỐ ĐỊNH RANDOM SEED — ĐẢM BẢO KẾT QUẢ TÁI LẬP
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from src.data_loader import fetch_and_prepare_data, format_vn, get_realtime_usd_vnd_rate
from src.features import DataTransformer
from src.ai_models import build_xgboost_optimized, build_transformer

USD_TO_VND = get_realtime_usd_vnd_rate()
print(f"💵 [TỶ GIÁ] Sử dụng tỷ giá USD/VND realtime: {format_vn(USD_TO_VND)} VNĐ\n")

LOOKBACK_WINDOW = 45

def evaluate_predictions(y_true, y_pred, model_name, ticker):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"=== KẾT QUẢ MÔ HÌNH {model_name.upper()} ===")
    if "VNM" in ticker.upper():
        print(f"❌ Sai số RMSE: {format_vn(rmse)} VNĐ")
        print(f"🎯 Sai số MAE : {format_vn(mae)} VNĐ (Lệch trung bình: {mape:.2f}%)\n")
    else:
        rmse_usd = rmse / USD_TO_VND
        mae_usd = mae / USD_TO_VND
        print(f"❌ Sai số RMSE: {format_vn(rmse)} VNĐ (tương đương ${rmse_usd:.2f} USD)")
        print(f"🎯 Sai số MAE : {format_vn(mae)} VNĐ (tương đương ${mae_usd:.2f} USD - Lệch trung bình: {mape:.2f}%)\n")
        
    return rmse, mae, mape

def cosine_decay(epoch):
    initial_lrate = 1e-4
    epochs = 100
    cos_outer = math.pi * epoch / epochs
    lrate = initial_lrate * 0.5 * (1.0 + math.cos(cos_outer))
    return max(lrate, 1e-5)

def log_prediction_to_file(ticker, last_date, last_close, risk_level, risk_ratio, xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper, rate_today=None):
    logs_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "predictions_history.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_trend_indicator(pred_val, ref_close):
        diff = pred_val - ref_close
        pct = (diff / ref_close) * 100
        emoji = "📈 TĂNG" if diff >= 0 else "📉 GIẢM"
        return f"({emoji} {pct:+.2f}%)"

    # Để đảm bảo tương thích 100% với Web App đọc file log, 
    # nhãn hiển thị là "XGBoost" nhưng giá trị thực tế ghi log là dự báo của mô hình Hybrid
    xgb_trend = get_trend_indicator(xgb_val, last_close)
    trans_trend = get_trend_indicator(trans_val, last_close)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== BẢN GHI DỰ BÁO ({timestamp}) ===\n")
        f.write(f"Mã chứng khoán: {ticker}\n")
        if "VNM" in ticker.upper():
            f.write(f"Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ\n")
            f.write(f"Rủi ro biến động: {risk_level} (Tỷ lệ: {risk_ratio:.2f}%)\n")
            f.write(f"Dự báo XGBoost: {format_vn(xgb_val)} VNĐ {xgb_trend} | Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ\n")
            f.write(f"Dự báo Transformer: {format_vn(trans_val)} VNĐ {trans_trend} | Khoảng an toàn: [{format_vn(trans_lower)} - {format_vn(trans_upper)}] VNĐ\n")
        else:
            rate = rate_today if rate_today else 25400
            f.write(f"Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ (${last_close/rate:,.2f} USD)\n")
            f.write(f"Rủi ro biến động: {risk_level} (Tỷ lệ: {risk_ratio:.2f}%)\n")
            f.write(f"Tỷ giá USD/VND quy đổi: 1 USD = {format_vn(rate)} VNĐ\n")
            f.write(f"Dự báo XGBoost: {format_vn(xgb_val)} VNĐ (${xgb_val/rate:.2f} USD) {xgb_trend} | Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ\n")
            f.write(f"Dự báo Transformer: {format_vn(trans_val)} VNĐ (${trans_val/rate:.2f} USD) {trans_trend} | Khoảng an toàn: [{format_vn(trans_lower)} - {format_vn(trans_upper)}] VNĐ\n")
        f.write("-" * 50 + "\n\n")

def main():
    print("🚀 [HỆ THỐNG] Khởi động Pipeline Huấn luyện Mô hình Lai (Hybrid Transformer-XGBoost)...")
    
    SENTIMENT_ENGINE = 'vader'
    TICKERS = ["VNM.VN", "GOOGL", "META"]
    
    # Cho phép chọn ticker cụ thể qua command line
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
            print(f"🎯 Chỉ chạy huấn luyện cho mã: {TICKERS[0]}")

    START_TRAIN = "2010-01-01"
    END_PREDICT = "2026-05-20"
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        LearningRateScheduler(cosine_decay, verbose=0)
    ]
    
    models_dir = os.path.join(ROOT_DIR, 'models')
    results_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    
    for ticker in TICKERS:
        print(f"\n--------------------------------------------------")
        print(f"🔄 BẮT ĐẦU HUẤN LUYỆN HYBRID CHO MÃ: {ticker}...")
        print(f"--------------------------------------------------")
        
        df = fetch_and_prepare_data(ticker, start_date=START_TRAIN, end_date=END_PREDICT, sentiment_engine=SENTIMENT_ENGINE)
        
        transformer = DataTransformer(time_steps=LOOKBACK_WINDOW)
        X_scaled, y_scaled, y_spread_scaled = transformer.fit_transform_data(df)
        X_3D, y_3D, y_spread_3D = transformer.create_sliding_windows(X_scaled, y_scaled, y_spread_scaled)
        
        # Chia tập dữ liệu 80/20 theo thời gian
        X_train_i, y_train_i, X_test_i, y_test_i, y_test_raw_i, y_train_spread_i, y_test_spread_i = transformer.split_train_test_chronological(df, X_3D, y_3D, y_spread_3D, train_ratio=0.8)
        
        # Tạo tập validation nhỏ 10% từ tập train phục vụ cho việc theo dõi khớp mạng Neural
        val_size = int(len(X_train_i) * 0.1)
        if val_size > 0:
            X_tr, y_tr = X_train_i[:-val_size], y_train_i[:-val_size]
            X_va, y_va = X_train_i[-val_size:], y_train_i[-val_size:]
            y_tr_spread = y_train_spread_i[:-val_size]
            y_va_spread = y_train_spread_i[-val_size:]
        else:
            X_tr, y_tr = X_train_i, y_train_i
            X_va, y_va = X_test_i, y_test_i
            y_tr_spread = y_train_spread_i
            y_va_spread = y_test_spread_i
        
        # Load best transformer parameters for the specific ticker if they exist
        best_params_ticker_path = os.path.join(ROOT_DIR, 'config', f'best_transformer_params_{ticker}.json')
        best_params_path = os.path.join(ROOT_DIR, 'config', 'best_transformer_params.json')
        
        d_model = 128
        num_heads = 8
        dropout_rate = 0.3
        learning_rate = 1e-4
        batch_size = 64
        
        chosen_params_path = best_params_ticker_path if os.path.exists(best_params_ticker_path) else best_params_path
        
        if os.path.exists(chosen_params_path):
            try:
                import json
                with open(chosen_params_path, 'r') as f:
                    best_params = json.load(f)
                d_model = best_params.get('d_model', d_model)
                num_heads = best_params.get('num_heads', num_heads)
                dropout_rate = best_params.get('dropout_rate', dropout_rate)
                learning_rate = best_params.get('learning_rate', learning_rate)
                batch_size = best_params.get('batch_size', batch_size)
                print(f"🥇 [TUNED] Đang sử dụng siêu tham số tối ưu từ file {os.path.basename(chosen_params_path)}:")
                print(f"   - d_model: {d_model}, num_heads: {num_heads}, dropout: {dropout_rate:.4f}, lr: {learning_rate:.6f}, batch: {batch_size}")
            except Exception as e:
                print(f"⚠️ Không thể đọc {chosen_params_path}: {e}")
        
        # 1. HUẤN LUYỆN CHÉO K-FOLD ĐỂ TRÍCH XUẤT OUT-OF-FOLD (OOF) EMBEDDINGS CHO XGBOOST
        print(f"🤖 [TRAIN] Đang tiến hành K-Fold Cross-Validation (K=5) để tạo đặc trưng OOF cho XGBoost...")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=False)
        
        X_train_latent_oof = np.zeros((len(X_train_i), 32))
        
        fold = 1
        for train_idx, val_idx in kf.split(X_train_i):
            print(f"   - Huấn luyện chéo Fold {fold}/5...")
            X_tr_fold, y_tr_fold = X_train_i[train_idx], y_train_i[train_idx]
            X_va_fold, y_va_fold = X_train_i[val_idx], y_train_i[val_idx]
            
            y_tr_spread_fold = y_train_spread_i[train_idx]
            y_va_spread_fold = y_train_spread_i[val_idx]
            
            fold_model = build_transformer(
                input_shape=(X_train_i.shape[1], X_train_i.shape[2]),
                d_model=d_model,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            fold_callbacks = [
                EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=0)
            ]
            
            fold_model.fit(
                X_tr_fold, 
                {"output_return": y_tr_fold, "output_spread": y_tr_spread_fold},
                validation_data=(
                    X_va_fold, 
                    {"output_return": y_va_fold, "output_spread": y_va_spread_fold}
                ),
                epochs=35,
                batch_size=batch_size,
                callbacks=fold_callbacks,
                verbose=0
            )
            
            fold_extractor = Model(inputs=fold_model.input, outputs=fold_model.get_layer("latent_embedding").output)
            X_train_latent_oof[val_idx] = fold_extractor.predict(X_va_fold, verbose=0)
            fold += 1
 
        # Huấn luyện mô hình Transformer chính (lưu trữ & dự báo live)
        print(f"🤖 [TRAIN] Đang huấn luyện Transformer chính cho riêng {ticker}...")
        transformer_model = build_transformer(
            input_shape=(X_tr.shape[1], X_tr.shape[2]),
            d_model=d_model,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        transformer_model.fit(
            X_tr, 
            {"output_return": y_tr, "output_spread": y_tr_spread},
            validation_data=(
                X_va, 
                {"output_return": y_va, "output_spread": y_va_spread}
            ),
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Thiết lập mô hình trích xuất đặc trưng chính
        print(f"🔍 [FEATURE EXTRACTION] Thiết lập bộ trích xuất đặc trưng ẩn chính...")
        feature_extractor = Model(
            inputs=transformer_model.input,
            outputs=transformer_model.get_layer("latent_embedding").output
        )
        
        # Lấy các chỉ báo kỹ thuật gốc ngày hiện tại cho toàn bộ tập Train
        X_train_today_all = X_train_i[:, -1, :]
        X_train_hybrid_all = np.concatenate([X_train_latent_oof, X_train_today_all], axis=1)
        
        # 2. HUẤN LUYỆN XGBOOST LAI (GIAI ĐOẠN 2)
        # Sử dụng 100% dữ liệu Train thô kết hợp đặc trưng OOF không rò rỉ dữ liệu để huấn luyện XGBoost
        print(f"🌳 [TRAIN] Đang huấn luyện mô hình lai XGBoost trên 100% dữ liệu OOF Train ({len(X_train_hybrid_all)} mẫu)...")
        xgb_model = build_xgboost_optimized(X_train_hybrid_all, y_train_i)
        
        # 3. ĐÁNH GIÁ KẾT QUẢ TRÊN TẬP KIỂM THỬ (TEST SET)
        split_idx = int(len(X_3D) * 0.8)
        df_align = df.iloc[transformer.time_steps:].reset_index(drop=True)
        # Căn chỉnh chỉ số bắt đầu sau Purge Gap (45 phiên)
        test_start_idx = split_idx + 45
        test_close_i = df_align.loc[test_start_idx:, 'close'].values[:len(X_test_i)]
        y_test_true_i = test_close_i * (1 + y_test_raw_i)
        
        # Trích xuất đặc trưng lai cho tập Test
        X_test_latent = feature_extractor.predict(X_test_i, verbose=0)
        X_test_today = X_test_i[:, -1, :]
        X_test_hybrid = np.concatenate([X_test_latent, X_test_today], axis=1)
        
        # Dự báo XGBoost lai (Hybrid)
        hybrid_xgb_scaled = xgb_model.predict(X_test_hybrid).reshape(-1, 1)
        hybrid_xgb_return = transformer.target_scaler.inverse_transform(hybrid_xgb_scaled).ravel()
        hybrid_xgb_preds = test_close_i * (1 + hybrid_xgb_return)
        
        # Dự báo Transformer gốc
        trans_preds_output = transformer_model.predict(X_test_i, verbose=0)
        trans_scaled = trans_preds_output[0] if isinstance(trans_preds_output, list) else trans_preds_output
        trans_return = transformer.target_scaler.inverse_transform(trans_scaled).ravel()
        trans_preds = test_close_i * (1 + trans_return)
        print(f"\n📊 BẢNG ĐÁNH GIÁ SAI SỐ CHO MÃ: {ticker}")
        evaluate_predictions(y_test_true_i, hybrid_xgb_preds, f"Hybrid XGBoost ({ticker})", ticker)
        evaluate_predictions(y_test_true_i, trans_preds, f"Transformer ({ticker})", ticker)
        
        # --- Trực quan hóa kết quả ---
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_true_i[-100:], label="Giá thực tế (VNĐ)", color='black', linewidth=2.5)
        plt.plot(hybrid_xgb_preds[-100:], label="Dự báo Hybrid XGBoost (VNĐ)", color='red', linestyle='--')
        plt.plot(trans_preds[-100:], label="Dự báo Transformer gốc (VNĐ)", color='blue', linestyle='-.')
        plt.ylabel("Giá (VNĐ)", fontsize=12)
        plt.title(f"XU HƯỚNG GIÁ THỰC TẾ VS DỰ BÁO CỦA {ticker} (MÔ HÌNH LAI HYBRID)", fontsize=14, fontweight='bold')
        plt.xlabel("100 Phiên giao dịch cuối cùng (Tập Test)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'model_battle_result_{ticker}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 Đã xuất biểu đồ cho {ticker} vào file: '{plot_path}'")
        
        # Lưu trữ các file mô hình có timestamp phiên bản và cập nhật bản latest
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        # Tên file phiên bản có timestamp
        xgb_ts_name = f'xgboost_model_{ticker}_{timestamp}.pkl'
        model_ts_name = f'transformer_model_{ticker}_{timestamp}.keras'
        feat_scaler_ts_name = f'feature_scaler_{ticker}_{timestamp}.pkl'
        targ_scaler_ts_name = f'target_scaler_{ticker}_{timestamp}.pkl'
        
        # Tên file latest mặc định
        xgb_latest_name = f'xgboost_model_{ticker}.pkl'
        model_latest_name = f'transformer_model_{ticker}.keras'
        feat_scaler_latest_name = f'feature_scaler_{ticker}.pkl'
        targ_scaler_latest_name = f'target_scaler_{ticker}.pkl'
        
        os.makedirs(models_dir, exist_ok=True)
        
        # 1. Lưu bản có timestamp phục vụ lưu vết lịch sử
        joblib.dump(xgb_model, os.path.join(models_dir, xgb_ts_name))
        transformer_model.save(os.path.join(models_dir, model_ts_name))
        joblib.dump(transformer.feature_scaler, os.path.join(models_dir, feat_scaler_ts_name))
        joblib.dump(transformer.target_scaler, os.path.join(models_dir, targ_scaler_ts_name))
        
        # 2. Lưu bản latest đè lên tệp cũ để backend API/inference hoạt động không đổi
        joblib.dump(xgb_model, os.path.join(models_dir, xgb_latest_name))
        transformer_model.save(os.path.join(models_dir, model_latest_name))
        joblib.dump(transformer.feature_scaler, os.path.join(models_dir, feat_scaler_latest_name))
        joblib.dump(transformer.target_scaler, os.path.join(models_dir, targ_scaler_latest_name))
        
        print(f"💾 Đã lưu Hybrid Models & Scalers phiên bản {timestamp} cho {ticker} (Bản latest được cập nhật thành công).")
        
        # --- DỰ BÁO CHO PHIÊN KẾ TIẾP (LIVE PREDICTION) ---
        print(f"\n🔮 BƯỚC 8: DỰ BÁO GIÁ MỞ CỬA KẾ TIẾP CHO {ticker}...")
        raw_df = yf.download(ticker, period="150d", progress=False)
        if not raw_df.empty:
            if type(raw_df.columns) == pd.MultiIndex:
                raw_df.columns = raw_df.columns.droplevel(1)
            raw_df.columns = [col.lower() for col in raw_df.columns]
            
            # Tải tỷ giá USD/VND trực tuyến (hoặc fallback)
            usd_vnd_rate = USD_TO_VND
            df_rate_hist = pd.DataFrame()
            try:
                df_rate_hist = yf.download("USDVND=X", period="150d", progress=False)
                if not df_rate_hist.empty:
                    if isinstance(df_rate_hist.columns, pd.MultiIndex):
                        df_rate_hist.columns = [col[0].lower() for col in df_rate_hist.columns]
                    else:
                        df_rate_hist.columns = [col.lower() for col in df_rate_hist.columns]
                    df_rate_hist = df_rate_hist[['close']].rename(columns={'close': 'rate_close'})
                    df_rate_hist['rate_close'] = df_rate_hist['rate_close'].apply(lambda x: x * 1000.0 if x < 1000.0 else x)
                    df_rate_hist.loc[(df_rate_hist['rate_close'] < 15000.0) | (df_rate_hist['rate_close'] > 28000.0), 'rate_close'] = np.nan
                    df_rate_hist['rate_close'] = df_rate_hist['rate_close'].ffill().bfill().fillna(USD_TO_VND)
                    usd_vnd_rate = float(df_rate_hist['rate_close'].iloc[-1])
            except Exception as e:
                print(f"  [WARNING] Không thể tải tỷ giá trực tuyến hôm nay: {e}")

            # Quy đổi giá trị sang VNĐ cho GOOGL/META
            if "VNM" not in ticker.upper():
                try:
                    if not df_rate_hist.empty:
                        raw_df = raw_df.merge(df_rate_hist, left_index=True, right_index=True, how='left')
                        raw_df['rate_close'] = raw_df['rate_close'].ffill().bfill().fillna(USD_TO_VND)
                        price_cols = ['open', 'high', 'low', 'close']
                        for col in price_cols:
                            raw_df[col] = raw_df[col] * raw_df['rate_close']
                        raw_df = raw_df.drop(columns=['rate_close'])
                        print(f"  [QUY ĐỔI] Đã quy đổi giá trị {ticker} sang VNĐ bằng tỷ giá động trực tuyến.")
                    else:
                        raise ValueError("Rỗng")
                except Exception as e:
                    print(f"  [WARNING] Lỗi quy đổi tỷ giá: {e}. Dùng tỷ giá realtime: {format_vn(USD_TO_VND)}.")
                    price_cols = ['open', 'high', 'low', 'close']
                    for col in price_cols:
                        raw_df[col] = raw_df[col] * USD_TO_VND

            # Tải chỉ số thị trường vĩ mô tương ứng để dự báo
            index_ticker = "VNM" if "VNM" in ticker.upper() else "^GSPC"
            try:
                start_idx_date = raw_df.index.min().strftime('%Y-%m-%d')
                end_idx_date = (raw_df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                raw_index = yf.download(index_ticker, start=start_idx_date, end=end_idx_date, progress=False)
                if not raw_index.empty:
                    df_index = raw_index.reset_index()
                    if isinstance(df_index.columns, pd.MultiIndex):
                        df_index.columns = [str(col[0]).lower() for col in df_index.columns]
                    else:
                        df_index.columns = [str(col).lower() for col in df_index.columns]
                    df_index['date'] = pd.to_datetime(df_index['date'])
                    df_index['market_return'] = df_index['close'].pct_change()
                    df_index = df_index[['date', 'market_return']]
                else:
                    df_index = pd.DataFrame(columns=['date', 'market_return'])
            except Exception as e:
                print(f"  [WARNING] Không thể tải dữ liệu chỉ số dự báo {index_ticker}: {e}")
                df_index = pd.DataFrame(columns=['date', 'market_return'])

            # Tải thêm chỉ số hoảng sợ vĩ mô VIX
            try:
                raw_vix = yf.download("^VIX", start=start_idx_date, end=end_idx_date, progress=False)
                if not raw_vix.empty:
                    df_vix = raw_vix.reset_index()
                    if isinstance(df_vix.columns, pd.MultiIndex):
                        df_vix.columns = [str(col[0]).lower() for col in df_vix.columns]
                    else:
                        df_vix.columns = [str(col).lower() for col in df_vix.columns]
                    df_vix['date'] = pd.to_datetime(df_vix['date'])
                    df_vix = df_vix[['date', 'close']].rename(columns={'close': 'vix'})
                else:
                    df_vix = pd.DataFrame(columns=['date', 'vix'])
            except Exception as e:
                print(f"  [WARNING] Không thể tải dữ liệu VIX: {e}")
                df_vix = pd.DataFrame(columns=['date', 'vix'])

            # Tải thêm lợi suất trái phiếu chính phủ Mỹ 10 năm ^TNX
            try:
                raw_tnx = yf.download("^TNX", start=start_idx_date, end=end_idx_date, progress=False)
                if not raw_tnx.empty:
                    df_tnx = raw_tnx.reset_index()
                    if isinstance(df_tnx.columns, pd.MultiIndex):
                        df_tnx.columns = [str(col[0]).lower() for col in df_tnx.columns]
                    else:
                        df_tnx.columns = [str(col).lower() for col in df_tnx.columns]
                    df_tnx['date'] = pd.to_datetime(df_tnx['date'])
                    df_tnx = df_tnx[['date', 'close']].rename(columns={'close': 'bond_yield_10y'})
                else:
                    df_tnx = pd.DataFrame(columns=['date', 'bond_yield_10y'])
            except Exception as e:
                print(f"  [WARNING] Không thể tải dữ liệu TNX: {e}")
                df_tnx = pd.DataFrame(columns=['date', 'bond_yield_10y'])

            # Tải chỉ số Dollar Index DX-Y.NYB
            try:
                raw_dxy = yf.download("DX-Y.NYB", start=start_idx_date, end=end_idx_date, progress=False)
                if not raw_dxy.empty:
                    df_dxy = raw_dxy.reset_index()
                    if isinstance(df_dxy.columns, pd.MultiIndex):
                        df_dxy.columns = [str(col[0]).lower() for col in df_dxy.columns]
                    else:
                        df_dxy.columns = [str(col).lower() for col in df_dxy.columns]
                    df_dxy['date'] = pd.to_datetime(df_dxy['date'])
                    df_dxy['dollar_index_change'] = df_dxy['close'].pct_change()
                    df_dxy = df_dxy[['date', 'dollar_index_change']]
                else:
                    df_dxy = pd.DataFrame(columns=['date', 'dollar_index_change'])
            except Exception as e:
                print(f"  [WARNING] Không thể tải dữ liệu DXY: {e}")
                df_dxy = pd.DataFrame(columns=['date', 'dollar_index_change'])

            # Gộp market_return, vix, tnx, dxy
            raw_df = raw_df.reset_index()
            raw_df.columns = [col.lower() for col in raw_df.columns]
            raw_df['date'] = pd.to_datetime(raw_df['date'])
            raw_df = pd.merge(raw_df, df_index, on='date', how='left')
            raw_df['market_return'] = raw_df['market_return'].fillna(0.0)
            raw_df = pd.merge(raw_df, df_vix, on='date', how='left')
            raw_df['vix'] = raw_df['vix'].ffill().bfill().fillna(20.0)
            raw_df = pd.merge(raw_df, df_tnx, on='date', how='left')
            raw_df['bond_yield_10y'] = raw_df['bond_yield_10y'].ffill().bfill().fillna(4.0)
            raw_df = pd.merge(raw_df, df_dxy, on='date', how='left')
            raw_df['dollar_index_change'] = raw_df['dollar_index_change'].fillna(0.0)
            
            # Tích hợp đặc trưng cảm xúc tin tức
            try:
                from src.news_sentiment import get_news_sentiment_features
                df_sent_pred = get_news_sentiment_features(ticker, raw_df['date'].dt.strftime('%Y-%m-%d').tolist(), engine=SENTIMENT_ENGINE)
                df_sent_pred['date'] = pd.to_datetime(df_sent_pred['date'])
                raw_df = pd.merge(raw_df, df_sent_pred, on='date', how='left')
            except Exception as e:
                print(f"  [WARNING] Không thể tích hợp tin tức dự báo: {e}")
                raw_df['sentiment_score'] = 0.0
                raw_df['news_volume'] = 0.0
            
            raw_df['sentiment_score'] = raw_df['sentiment_score'].fillna(0.0)
            raw_df['news_volume'] = raw_df['news_volume'].fillna(0.0)
            raw_df = raw_df.set_index('date')
            
            # Áp dụng Kalman Filter để làm mịn giá đóng cửa
            try:
                from src.features import kalman_filter
            except ImportError:
                from features import kalman_filter
            raw_df['close_smoothed'] = kalman_filter(raw_df['close'])
            
            # Tính các chỉ báo kỹ thuật dựa trên close_smoothed
            raw_df['rsi_14'] = ta.rsi(raw_df['close_smoothed'], length=14)
            raw_df['rsi_lag1'] = raw_df['rsi_14'].shift(1)
            macd_df = ta.macd(raw_df['close_smoothed'], fast=12, slow=26, signal=9)
            raw_df = pd.concat([raw_df, macd_df], axis=1)
            
            macd_cols_to_drop = [col for col in macd_df.columns if 'MACDh' in col or 'MACDs' in col]
            raw_df = raw_df.drop(columns=macd_cols_to_drop)

            raw_df['volatility_20'] = raw_df['close_smoothed'].pct_change().rolling(window=20).std()
            raw_df['close_lag1'] = raw_df['close_smoothed'].shift(1)
            raw_df['close_lag2'] = raw_df['close_smoothed'].shift(2)
            raw_df['close_lag3'] = raw_df['close_smoothed'].shift(3)
            raw_df['open_lag1'] = raw_df['open'].shift(1)
            raw_df['open_lag2'] = raw_df['open'].shift(2)
            raw_df['volume_change'] = raw_df['volume'].pct_change()
            raw_df['intraday_return'] = (raw_df['close'] - raw_df['open']) / raw_df['open']
            
            bb_df = ta.bbands(raw_df['close_smoothed'], length=20, std=2)
            raw_df['bb_lower'] = bb_df.iloc[:, 0]
            raw_df['bb_middle'] = bb_df.iloc[:, 1]
            raw_df['bb_upper'] = bb_df.iloc[:, 2]
            raw_df['atr_14'] = ta.atr(raw_df['high'], raw_df['low'], raw_df['close_smoothed'], length=14)
            raw_df['ema_14'] = ta.ema(raw_df['close_smoothed'], length=14)
            raw_df['roc_10'] = ta.roc(raw_df['close_smoothed'], length=10)
            adx_raw = ta.adx(raw_df['high'], raw_df['low'], raw_df['close_smoothed'], length=14)
            raw_df['adx_14'] = adx_raw.iloc[:, 0]
            
            # Clean and calculate stationary features using DataTransformer
            raw_df_reset = raw_df.reset_index()
            raw_df_transformed = transformer.transform_df(raw_df_reset)
            
            recent_features = raw_df_transformed.tail(LOOKBACK_WINDOW)
            
            if len(recent_features) == LOOKBACK_WINDOW:
                recent_scaled = transformer.feature_scaler.transform(recent_features.values)
                X_predict = recent_scaled.reshape(1, LOOKBACK_WINDOW, len(transformer.feature_cols))
                
                # Trích xuất đặc trưng lai cho Live Prediction (32 embedding + 22 raw indicators)
                X_predict_latent = feature_extractor.predict(X_predict, verbose=0)
                X_predict_today = X_predict[0, -1, :].reshape(1, -1)
                X_predict_hybrid = np.concatenate([X_predict_latent, X_predict_today], axis=1)
                
                # Dự báo từ XGBoost Lai
                xgb_pred_scaled = xgb_model.predict(X_predict_hybrid).reshape(-1, 1)
                xgb_return_future = transformer.target_scaler.inverse_transform(xgb_pred_scaled)[0][0]
                
                trans_pred_output = transformer_model.predict(X_predict, verbose=0)
                trans_pred_scaled = trans_pred_output[0] if isinstance(trans_pred_output, list) else trans_pred_output
                trans_return_future = transformer.target_scaler.inverse_transform(trans_pred_scaled)[0][0]
                
                last_close = raw_df['close'].iloc[-1]

                last_date = raw_df.index[-1].strftime('%Y-%m-%d')
                
                xgb_val = last_close * (1 + xgb_return_future)
                trans_val = last_close * (1 + trans_return_future)
                
                last_atr = raw_df['atr_14'].iloc[-1]
                risk_ratio = (last_atr / last_close) * 100
                if risk_ratio < 1.5:
                    risk_level = "Thấp (An toàn - Thị trường ổn định) 🟢"
                elif risk_ratio < 3.0:
                    risk_level = "Trung bình (Biến động nhẹ - Thận trọng) 🟡"
                else:
                    risk_level = "Cao (Nguy hiểm - Biến động cực mạnh / Có Drama hoặc Tin tức lớn) 🔴 [CẢNH BÁO: HẠN CHẾ ĐẶT LỆNH KHỚP NGAY]"

                xgb_lower = xgb_val - 1.5 * last_atr
                xgb_upper = xgb_val + 1.5 * last_atr
                trans_lower = trans_val - 1.5 * last_atr
                trans_upper = trans_val + 1.5 * last_atr

                xgb_trend = f"📈 TĂNG ({((xgb_val - last_close)/last_close)*100:+.2f}%)" if xgb_val >= last_close else f"📉 GIẢM ({((xgb_val - last_close)/last_close)*100:+.2f}%)"
                trans_trend = f"📈 TĂNG ({((trans_val - last_close)/last_close)*100:+.2f}%)" if trans_val >= last_close else f"📉 GIẢM ({((trans_val - last_close)/last_close)*100:+.2f}%)"

                if "VNM" in ticker.upper():
                    print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {format_vn(last_close)} VNĐ")
                    print(f"⚠️ Mức độ rủi ro biến động hiện tại: {risk_level} (Tỷ lệ biến động: {risk_ratio:.2f}%)")
                    print(f"🔮 DỰ BÁO GIÁ MỞ CỬA & KHOẢNG AN TOÀN CHO PHIÊN KẾ TIẾP:")
                    print(f"  - 🌳 Hybrid XGBoost: {format_vn(xgb_val)} VNĐ | {xgb_trend} | Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ")
                    print(f"  - 🤖 Transformer   : {format_vn(trans_val)} VNĐ | {trans_trend} | Khoảng an toàn: [{format_vn(trans_lower)} - {format_vn(trans_upper)}] VNĐ")
                    log_prediction_to_file(ticker, last_date, last_close, risk_level, risk_ratio, xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper)
                else:
                    rate_today = usd_vnd_rate
                    print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {format_vn(last_close)} VNĐ (tương đương ${last_close/rate_today:,.2f} USD)")
                    print(f"⚠️ Mức độ rủi ro biến động hiện tại: {risk_level} (Tỷ lệ biến động: {risk_ratio:.2f}%)")
                    print(f"🔮 DỰ BÁO GIÁ MỞ CỬA & KHOẢNG AN TOÀN CHO PHIÊN KẾ TIẾP (tỷ giá quy đổi: 1 USD = {format_vn(rate_today)} VNĐ):")
                    print(f"  - 🌳 Hybrid XGBoost: {format_vn(xgb_val)} VNĐ (tương đương ${xgb_val/rate_today:.2f} USD) | {xgb_trend}")
                    print(f"                       Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ (tương đương ${xgb_lower/rate_today:.2f} - ${xgb_upper/rate_today:.2f} USD)")
                    print(f"  - 🤖 Transformer   : {format_vn(trans_val)} VNĐ (tương đương ${trans_val/rate_today:.2f} USD) | {trans_trend}")
                    print(f"                       Khoảng an toàn: [{format_vn(trans_lower)} - {format_vn(trans_upper)}] VNĐ (tương đương ${trans_lower/rate_today:.2f} - ${trans_upper/rate_today:.2f} USD)")
                    log_prediction_to_file(ticker, last_date, last_close, risk_level, risk_ratio, xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper, rate_today)
            else:
                print(f"Lỗi: Không đủ dữ liệu {LOOKBACK_WINDOW} ngày cho {ticker}")
        else:
            print(f"Lỗi: Không tải được dữ liệu trực tuyến cho {ticker}")
        print(f"--------------------------------------------------\n")

if __name__ == "__main__":
    main()
