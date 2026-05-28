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
    # Tính Phần trăm lệch trung bình tuyệt đối (MAPE) để thấy độ chính xác tương đối
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"=== KẾT QUẢ MÔ HÌNH {model_name.upper()} ===")
    if "VNM" in ticker.upper():
        print(f"❌ Sai số RMSE: {format_vn(rmse)} VNĐ")
        print(f"🎯 Sai số MAE : {format_vn(mae)} VNĐ (Lệch trung bình: {mape:.2f}%)\n")
    else:
        # Đối với GOOGL và META, hiển thị cả VNĐ và quy đổi ngược lại USD
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
    print("🚀 [HỆ THỐNG] Khởi động Pipeline Huấn luyện (Môi trường Đa mã Quy đổi VNĐ)...")
    
    # Cấu hình Sentiment Engine: 'finbert' (Độ chính xác cao - Nặng) hoặc 'vader' (Nhẹ - Nhanh)
    # FinBERT tự động tải mô hình từ HuggingFace khoảng 400MB khi chạy lần đầu tiên.
    SENTIMENT_ENGINE = 'finbert'
    
    TICKERS = ["VNM.VN", "GOOGL", "META"]
    START_TRAIN = "2010-01-01"
    END_PREDICT = "2026-05-20"
    
    # 💡 CẤU HÌNH QUAN TRỌNG: Đặt INDIVIDUAL_TRAINING = True để huấn luyện riêng cho từng mã.
    # Huấn luyện riêng biệt giúp mô hình học chính xác đặc thù của từng loại cổ phiếu, giảm sai số đáng kể.
    INDIVIDUAL_TRAINING = True 
    
    # Cấu hình callbacks cho mạng Neural (sử dụng Cosine Decay và Plateau Decay điều phối học tập)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        LearningRateScheduler(cosine_decay, verbose=0)
    ]
    
    models_dir = os.path.join(ROOT_DIR, 'models')
    results_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    
    if INDIVIDUAL_TRAINING:
        print("\n" + "="*60)
        print("🎯 PHƯƠNG ÁN 1: HUẤN LUYỆN ĐỘC LẬP CHO TỪNG CỔ PHIẾU (TỐI ƯU SAI SỐ)")
        print("="*60 + "\n")
        
        for ticker in TICKERS:
            print(f"\n--------------------------------------------------")
            print(f"🔄 BẮT ĐẦU HUẤN LUYỆN RIÊNG CHO MÃ: {ticker}...")
            print(f"--------------------------------------------------")
            
            df = fetch_and_prepare_data(ticker, start_date=START_TRAIN, end_date=END_PREDICT, sentiment_engine=SENTIMENT_ENGINE)
            
            transformer = DataTransformer(time_steps=LOOKBACK_WINDOW)
            X_scaled, y_scaled = transformer.fit_transform_data(df)
            X_3D, y_3D = transformer.create_sliding_windows(X_scaled, y_scaled)
            
            # Chia tập dữ liệu 80/20 theo thời gian
            X_train_i, y_train_i, X_test_i, y_test_i, y_test_raw_i = transformer.split_train_test_chronological(df, X_3D, y_3D, train_ratio=0.8)
            
            # Tạo tập validation nhỏ 10% từ tập train phục vụ cho việc theo dõi khớp mạng Neural
            val_size = int(len(X_train_i) * 0.1)
            if val_size > 0:
                X_tr, y_tr = X_train_i[:-val_size], y_train_i[:-val_size]
                X_va, y_va = X_train_i[-val_size:], y_train_i[-val_size:]
            else:
                X_tr, y_tr = X_train_i, y_train_i
                X_va, y_va = X_test_i, y_test_i
            
            # 1. Huấn luyện XGBoost
            print(f"🌳 [TRAIN] Đang tối ưu XGBoost cho riêng {ticker}...")
            X_train_flat = X_tr.reshape(X_tr.shape[0], -1)
            xgb_model = build_xgboost_optimized(X_train_flat, y_tr)
            
            # 2. Huấn luyện Transformer
            print(f"🤖 [TRAIN] Đang huấn luyện Transformer cho riêng {ticker}...")
            transformer_model = build_transformer(input_shape=(X_tr.shape[1], X_tr.shape[2]))
            transformer_model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            # Đánh giá kết quả trên tập kiểm thử (test) của mã này
            X_test_flat_i = X_test_i.reshape(X_test_i.shape[0], -1)
            split_idx = int(len(X_3D) * 0.8)
            df_align = df.iloc[transformer.time_steps:].reset_index(drop=True)
            test_close_i = df_align.loc[split_idx:, 'close'].values[:len(X_test_i)]
            y_test_true_i = test_close_i * (1 + y_test_raw_i)
            
            # Dự báo XGBoost
            xgb_scaled = xgb_model.predict(X_test_flat_i).reshape(-1, 1)
            xgb_return = transformer.target_scaler.inverse_transform(xgb_scaled).ravel()
            xgb_preds = test_close_i * (1 + xgb_return)
            
            # Dự báo Transformer
            trans_scaled = transformer_model.predict(X_test_i, verbose=0)
            trans_return = transformer.target_scaler.inverse_transform(trans_scaled).ravel()
            trans_preds = test_close_i * (1 + trans_return)
            
            print(f"\n📊 BẢNG ĐÁNH GIÁ SAI SỐ CHO MÃ: {ticker}")
            evaluate_predictions(y_test_true_i, xgb_preds, f"XGBoost ({ticker})", ticker)
            evaluate_predictions(y_test_true_i, trans_preds, f"Transformer ({ticker})", ticker)
            
            # --- Trực quan hóa kết quả cho mã này ---
            plt.figure(figsize=(15, 7))
            if "VNM" in ticker.upper():
                plt.plot(y_test_true_i[-100:], label="Giá thực tế (VNĐ)", color='black', linewidth=2.5)
                plt.plot(xgb_preds[-100:], label="Dự báo XGBoost (VNĐ)", color='red', linestyle='--')
                plt.plot(trans_preds[-100:], label="Dự báo Transformer (VNĐ)", color='blue', linestyle='-.')
                plt.ylabel("Giá (VNĐ)", fontsize=12)
            else:
                plt.plot(y_test_true_i[-100:], label="Giá thực tế VNĐ", color='black', linewidth=2.5)
                plt.plot(xgb_preds[-100:], label="Dự báo XGBoost VNĐ", color='red', linestyle='--')
                plt.plot(trans_preds[-100:], label="Dự báo Transformer VNĐ", color='blue', linestyle='-.')
                plt.ylabel("Giá quy đổi (VNĐ)", fontsize=12)
                
            plt.title(f"XU HƯỚNG GIÁ THỰC TẾ VS DỰ BÁO CỦA {ticker} (HUẤN LUYỆN ĐỘC LẬP)", fontsize=14, fontweight='bold')
            plt.xlabel("100 Phiên giao dịch cuối cùng (Tập Test)", fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'model_battle_result_{ticker}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"💾 Đã xuất biểu đồ cho {ticker} vào file: '{plot_path}'")
            
            # Lưu mô hình độc lập cho mã này
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(xgb_model, os.path.join(models_dir, f'xgboost_model_{ticker}.pkl'))
            transformer_model.save(os.path.join(models_dir, f'transformer_model_{ticker}.keras'))
            joblib.dump(transformer.feature_scaler, os.path.join(models_dir, f'feature_scaler_{ticker}.pkl'))
            joblib.dump(transformer.target_scaler, os.path.join(models_dir, f'target_scaler_{ticker}.pkl'))
            
            # --- DỰ BÁO CHO PHIÊN KẾ TIẾP CỦA MÃ NÀY ---
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
                        # Chuẩn hóa tỷ giá nếu < 1000 (yfinance lưu dạng 19.5 thay vì 19500)
                        df_rate_hist['rate_close'] = df_rate_hist['rate_close'].apply(lambda x: x * 1000.0 if x < 1000.0 else x)
                        # Loại bỏ các giá trị nhiễu ngoài khoảng tỷ giá USD/VND lịch sử hợp lý [15000, 28000]
                        df_rate_hist.loc[(df_rate_hist['rate_close'] < 15000.0) | (df_rate_hist['rate_close'] > 28000.0), 'rate_close'] = np.nan
                        df_rate_hist['rate_close'] = df_rate_hist['rate_close'].ffill().bfill().fillna(USD_TO_VND)
                        usd_vnd_rate = float(df_rate_hist['rate_close'].iloc[-1])
                except Exception as e:
                    print(f"  [WARNING] Khong the tai ty gia truc tuyen hom nay: {e}")

                # Quy đổi giá trị từ USD sang VNĐ cho GOOGL/META sử dụng tỷ giá động theo từng ngày
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
                        print(f"  [WARNING] Loi quy doi ty gia: {e}. Dung ty gia realtime: {format_vn(USD_TO_VND)}.")
                        price_cols = ['open', 'high', 'low', 'close']
                        for col in price_cols:
                            raw_df[col] = raw_df[col] * USD_TO_VND

                # Tải thêm chỉ số thị trường vĩ mô tương ứng để dự báo
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
                    print(f"  [WARNING] Khong the tai du lieu chi so du bao {index_ticker}: {e}")
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
                    print(f"  [WARNING] Khong the tai du lieu VIX: {e}")
                    df_vix = pd.DataFrame(columns=['date', 'vix'])

                # Gộp market_return và vix vào raw_df
                raw_df = raw_df.reset_index()
                raw_df.columns = [col.lower() for col in raw_df.columns]
                raw_df['date'] = pd.to_datetime(raw_df['date'])
                raw_df = pd.merge(raw_df, df_index, on='date', how='left')
                raw_df['market_return'] = raw_df['market_return'].fillna(0.0)
                raw_df = pd.merge(raw_df, df_vix, on='date', how='left')
                raw_df['vix'] = raw_df['vix'].ffill().bfill().fillna(20.0)
                
                # Tích hợp đặc trưng phân tích cảm xúc tin tức để dự báo
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
                

                # Tính các chỉ báo
                raw_df['rsi_14'] = ta.rsi(raw_df['close'], length=14)
                raw_df['rsi_lag1'] = raw_df['rsi_14'].shift(1)
                macd_df = ta.macd(raw_df['close'], fast=12, slow=26, signal=9)
                raw_df = pd.concat([raw_df, macd_df], axis=1)
                
                # Drop unwanted macd columns
                macd_cols_to_drop = [col for col in macd_df.columns if 'MACDh' in col or 'MACDs' in col]
                raw_df = raw_df.drop(columns=macd_cols_to_drop)

                raw_df['volatility_20'] = raw_df['close'].pct_change().rolling(window=20).std()
                raw_df['close_lag1'] = raw_df['close'].shift(1)
                raw_df['close_lag2'] = raw_df['close'].shift(2)
                raw_df['close_lag3'] = raw_df['close'].shift(3)
                raw_df['open_lag1'] = raw_df['open'].shift(1)
                raw_df['open_lag2'] = raw_df['open'].shift(2)
                raw_df['volume_change'] = raw_df['volume'].pct_change()
                raw_df['intraday_return'] = (raw_df['close'] - raw_df['open']) / raw_df['open']
                
                bb_df = ta.bbands(raw_df['close'], length=20, std=2)
                raw_df['bb_lower'] = bb_df.iloc[:, 0]
                raw_df['bb_middle'] = bb_df.iloc[:, 1]
                raw_df['bb_upper'] = bb_df.iloc[:, 2]
                raw_df['atr_14'] = ta.atr(raw_df['high'], raw_df['low'], raw_df['close'], length=14)
                
                # Tính bổ sung EMA, ROC, ADX
                raw_df['ema_14'] = ta.ema(raw_df['close'], length=14)
                raw_df['roc_10'] = ta.roc(raw_df['close'], length=10)
                adx_raw = ta.adx(raw_df['high'], raw_df['low'], raw_df['close'], length=14)
                raw_df['adx_14'] = adx_raw.iloc[:, 0]
                
                recent_features = raw_df[transformer.feature_cols].dropna().tail(LOOKBACK_WINDOW)
                
                if len(recent_features) == LOOKBACK_WINDOW:
                    recent_scaled = transformer.feature_scaler.transform(recent_features.values)
                    X_predict = recent_scaled.reshape(1, LOOKBACK_WINDOW, len(transformer.feature_cols))
                    
                    # Dự báo
                    xgb_pred_scaled = xgb_model.predict(X_predict.reshape(1, -1)).reshape(-1, 1)
                    xgb_return_future = transformer.target_scaler.inverse_transform(xgb_pred_scaled)[0][0]
                    
                    trans_pred_scaled = transformer_model.predict(X_predict, verbose=0)
                    trans_return_future = transformer.target_scaler.inverse_transform(trans_pred_scaled)[0][0]
                    
                    last_close = recent_features['close'].iloc[-1]
                    last_date = recent_features.index[-1].strftime('%Y-%m-%d')
                    
                    xgb_val = last_close * (1 + xgb_return_future)
                    trans_val = last_close * (1 + trans_return_future)
                    
                    # Tính toán mức độ rủi ro biến động dựa trên ATR và khoảng giá an toàn
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
                        print(f"  - 🌳 XGBoost    : {format_vn(xgb_val)} VNĐ | {xgb_trend} | Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ")
                        print(f"  - 🤖 Transformer: {format_vn(trans_val)} VNĐ | {trans_trend} | Khoảng an toàn: [{format_vn(trans_lower)} - {format_vn(trans_upper)}] VNĐ")
                        log_prediction_to_file(ticker, last_date, last_close, risk_level, risk_ratio, xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper)
                    else:
                        rate_today = usd_vnd_rate
                        print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {format_vn(last_close)} VNĐ (tương đương ${last_close/rate_today:,.2f} USD)")
                        print(f"⚠️ Mức độ rủi ro biến động hiện tại: {risk_level} (Tỷ lệ biến động: {risk_ratio:.2f}%)")
                        print(f"🔮 DỰ BÁO GIÁ MỞ CỬA & KHOẢNG AN TOÀN CHO PHIÊN KẾ TIẾP (tỷ giá quy đổi: 1 USD = {format_vn(rate_today)} VNĐ):")
                        print(f"  - 🌳 XGBoost    : {format_vn(xgb_val)} VNĐ (tương đương ${xgb_val/rate_today:.2f} USD) | {xgb_trend}")
                        print(f"                    Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ (tương đương ${xgb_lower/rate_today:.2f} - ${xgb_upper/rate_today:.2f} USD)")
                        print(f"  - 🤖 Transformer: {format_vn(trans_val)} VNĐ (tương đương ${trans_val/rate_today:.2f} USD) | {trans_trend}")
                        print(f"                    Khoảng an toàn: [{format_vn(trans_lower)} - {format_vn(trans_upper)}] VNĐ (tương đương ${trans_lower/rate_today:.2f} - ${trans_upper/rate_today:.2f} USD)")
                        log_prediction_to_file(ticker, last_date, last_close, risk_level, risk_ratio, xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper, rate_today)
                else:
                    print(f"Lỗi: Không đủ dữ liệu {LOOKBACK_WINDOW} ngày cho {ticker}")
            else:
                print(f"Lỗi: Không tải được dữ liệu trực tuyến cho {ticker}")
            print(f"--------------------------------------------------\n")
            
    else:
        # ==========================================
        # PHƯƠNG ÁN 2: HUẤN LUYỆN GỘP CHUNG (GLOBAL TRAINING)
        # ==========================================
        print("\n" + "="*60)
        print("📊 PHƯƠNG ÁN 2: HUẤN LUYỆN GỘP DỮ LIỆU ĐA MÃ CHUNG")
        print("="*60 + "\n")
        
        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []
        transformers = {}
        test_data = {}
        
        for ticker in TICKERS:
            print(f"🔄 Đang chuẩn bị dữ liệu (VNĐ) cho mã: {ticker}...")
            df = fetch_and_prepare_data(ticker, start_date=START_TRAIN, end_date=END_PREDICT, sentiment_engine=SENTIMENT_ENGINE)
            transformer = DataTransformer(time_steps=LOOKBACK_WINDOW)
            X_scaled, y_scaled = transformer.fit_transform_data(df)
            X_3D, y_3D = transformer.create_sliding_windows(X_scaled, y_scaled)
            
            X_train_i, y_train_i, X_test_i, y_test_i, y_test_raw_i = transformer.split_train_test_chronological(df, X_3D, y_3D, train_ratio=0.8)
            X_train_list.append(X_train_i)
            y_train_list.append(y_train_i)
            X_val_list.append(X_test_i)
            y_val_list.append(y_test_i)
            transformers[ticker] = transformer
            
            split_idx = int(len(X_3D) * 0.8)
            df_align = df.iloc[transformer.time_steps:].reset_index(drop=True)
            test_close_prices_i = df_align.loc[split_idx:, 'close'].values[:len(X_test_i)]
            y_test_true_prices_i = test_close_prices_i * (1 + y_test_raw_i)
            
            test_data[ticker] = {
                'X_test': X_test_i,
                'y_test': y_test_i,
                'y_test_raw': y_test_raw_i,
                'test_close_prices': test_close_prices_i,
                'y_test_true_prices': y_test_true_prices_i
            }

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        
        # 1. Huấn luyện XGBoost chung
        print("🧠 [TRAIN] Đang huấn luyện mô hình XGBoost chung...")
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        xgb_model = build_xgboost_optimized(X_train_flat, y_train)
        
        # 2. Huấn luyện Transformer chung
        print("\n🧠 [TRAIN] Đang huấn luyện mạng Deep Learning Transformer chung...")
        transformer_model = build_transformer(input_shape=(X_train.shape[1], X_train.shape[2]))
        transformer_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64, callbacks=callbacks, verbose=1)

        print("\n" + "="*60)
        print("📊 BẢNG ĐÁNH GIÁ SAI SỐ CHI TIẾT TRÊN TỪNG CỔ PHIẾU (TẬP TEST GỘP)")
        print("="*60)
        
        for ticker in TICKERS:
            t_data = test_data[ticker]
            t_trans = transformers[ticker]
            X_test_i = t_data['X_test']
            X_test_flat_i = X_test_i.reshape(X_test_i.shape[0], -1)
            test_close_i = t_data['test_close_prices']
            y_test_true_i = t_data['y_test_true_prices']
            
            xgb_scaled = xgb_model.predict(X_test_flat_i).reshape(-1, 1)
            xgb_return = t_trans.target_scaler.inverse_transform(xgb_scaled).ravel()
            xgb_preds = test_close_i * (1 + xgb_return)
            
            trans_scaled = transformer_model.predict(X_test_i, verbose=0)
            trans_return = t_trans.target_scaler.inverse_transform(trans_scaled).ravel()
            trans_preds = test_close_i * (1 + trans_return)
            
            print(f"\n--- [ ĐÁNH GIÁ CHO MÃ: {ticker} ] ---")
            evaluate_predictions(y_test_true_i, xgb_preds, f"XGBoost ({ticker})", ticker)
            evaluate_predictions(y_test_true_i, trans_preds, f"Transformer ({ticker})", ticker)
            
            plt.figure(figsize=(15, 7))
            if "VNM" in ticker.upper():
                plt.plot(y_test_true_i[-100:], label="Giá thực tế (VNĐ)", color='black', linewidth=2.5)
                plt.plot(xgb_preds[-100:], label="Dự báo XGBoost (VNĐ)", color='red', linestyle='--')
                plt.plot(trans_preds[-100:], label="Dự báo Transformer (VNĐ)", color='blue', linestyle='-.')
                plt.ylabel("Giá (VNĐ)", fontsize=12)
            else:
                plt.plot(y_test_true_i[-100:], label="Giá thực tế VNĐ", color='black', linewidth=2.5)
                plt.plot(xgb_preds[-100:], label="Dự báo XGBoost VNĐ", color='red', linestyle='--')
                plt.plot(trans_preds[-100:], label="Dự báo Transformer VNĐ", color='blue', linestyle='-.')
                plt.ylabel("Giá quy đổi (VNĐ)", fontsize=12)
            plt.title(f"XU HƯỚNG GIÁ THỰC TẾ VS DỰ BÁO CỦA {ticker} (HUẤN LUYỆN GỘP CHUNG)", fontsize=14, fontweight='bold')
            plt.xlabel("100 Phiên giao dịch cuối cùng (Tập Test)", fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'model_battle_result_{ticker}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
        # Lưu mô hình gộp chung và toàn bộ scalers sau khi đánh giá xong
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_model.pkl'))
        transformer_model.save(os.path.join(models_dir, 'transformer_model.keras'))
        for t in TICKERS:
            joblib.dump(transformers[t].feature_scaler, os.path.join(models_dir, f'feature_scaler_{t}.pkl'))
            joblib.dump(transformers[t].target_scaler, os.path.join(models_dir, f'target_scaler_{t}.pkl'))
            
        print(f"\n🔮 BƯỚC 8: DỰ BÁO GIÁ MỞ CỬA KẾ TIẾP CHO TỪNG CỔ PHIẾU (DỮ LIỆU GỘP)...")
        # Dự báo với mô hình gộp
        for ticker in TICKERS:
            t_trans = transformers[ticker]
            print(f"\nĐang tải dữ liệu thời gian thực mới nhất cho {ticker}...")
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
                        # Chuẩn hóa tỷ giá nếu < 1000 (yfinance lưu dạng 19.5 thay vì 19500)
                        df_rate_hist['rate_close'] = df_rate_hist['rate_close'].apply(lambda x: x * 1000.0 if x < 1000.0 else x)
                        # Loại bỏ các giá trị nhiễu ngoài khoảng tỷ giá USD/VND lịch sử hợp lý [15000, 28000]
                        df_rate_hist.loc[(df_rate_hist['rate_close'] < 15000.0) | (df_rate_hist['rate_close'] > 28000.0), 'rate_close'] = np.nan
                        df_rate_hist['rate_close'] = df_rate_hist['rate_close'].ffill().bfill().fillna(USD_TO_VND)
                        usd_vnd_rate = float(df_rate_hist['rate_close'].iloc[-1])
                except Exception as e:
                    print(f"  [WARNING] Khong the tai ty gia truc tuyen hom nay: {e}")

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
                        print(f"  [WARNING] Loi quy doi ty gia: {e}. Dung ty gia realtime: {format_vn(USD_TO_VND)}.")
                        price_cols = ['open', 'high', 'low', 'close']
                        for col in price_cols:
                            raw_df[col] = raw_df[col] * USD_TO_VND

                # Tải thêm chỉ số thị trường vĩ mô tương ứng để dự báo
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
                    print(f"  [WARNING] Khong the tai du lieu chi so du bao {index_ticker}: {e}")
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
                    print(f"  [WARNING] Khong the tai du lieu VIX: {e}")
                    df_vix = pd.DataFrame(columns=['date', 'vix'])

                # Gộp market_return và vix vào raw_df
                raw_df = raw_df.reset_index()
                raw_df.columns = [col.lower() for col in raw_df.columns]
                raw_df['date'] = pd.to_datetime(raw_df['date'])
                raw_df = pd.merge(raw_df, df_index, on='date', how='left')
                raw_df['market_return'] = raw_df['market_return'].fillna(0.0)
                raw_df = pd.merge(raw_df, df_vix, on='date', how='left')
                raw_df['vix'] = raw_df['vix'].ffill().bfill().fillna(20.0)
                
                # Tích hợp đặc trưng phân tích cảm xúc tin tức để dự báo
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


                raw_df['rsi_14'] = ta.rsi(raw_df['close'], length=14)
                raw_df['rsi_lag1'] = raw_df['rsi_14'].shift(1)
                macd_df = ta.macd(raw_df['close'], fast=12, slow=26, signal=9)
                raw_df = pd.concat([raw_df, macd_df], axis=1)
                
                # Drop unwanted macd columns
                macd_cols_to_drop = [col for col in macd_df.columns if 'MACDh' in col or 'MACDs' in col]
                raw_df = raw_df.drop(columns=macd_cols_to_drop)

                raw_df['volatility_20'] = raw_df['close'].pct_change().rolling(window=20).std()
                raw_df['close_lag1'] = raw_df['close'].shift(1)
                raw_df['close_lag2'] = raw_df['close'].shift(2)
                raw_df['close_lag3'] = raw_df['close'].shift(3)
                raw_df['open_lag1'] = raw_df['open'].shift(1)
                raw_df['open_lag2'] = raw_df['open'].shift(2)
                raw_df['volume_change'] = raw_df['volume'].pct_change()
                raw_df['intraday_return'] = (raw_df['close'] - raw_df['open']) / raw_df['open']
                
                bb_df = ta.bbands(raw_df['close'], length=20, std=2)
                raw_df['bb_lower'] = bb_df.iloc[:, 0]
                raw_df['bb_middle'] = bb_df.iloc[:, 1]
                raw_df['bb_upper'] = bb_df.iloc[:, 2]
                raw_df['atr_14'] = ta.atr(raw_df['high'], raw_df['low'], raw_df['close'], length=14)
                
                # Tính bổ sung EMA, ROC, ADX
                raw_df['ema_14'] = ta.ema(raw_df['close'], length=14)
                raw_df['roc_10'] = ta.roc(raw_df['close'], length=10)
                adx_raw = ta.adx(raw_df['high'], raw_df['low'], raw_df['close'], length=14)
                raw_df['adx_14'] = adx_raw.iloc[:, 0]
                
                recent_features = raw_df[t_trans.feature_cols].dropna().tail(LOOKBACK_WINDOW)
                if len(recent_features) == LOOKBACK_WINDOW:
                    recent_scaled = t_trans.feature_scaler.transform(recent_features.values)
                    X_predict = recent_scaled.reshape(1, LOOKBACK_WINDOW, len(t_trans.feature_cols))
                    xgb_pred_scaled = xgb_model.predict(X_predict.reshape(1, -1)).reshape(-1, 1)
                    xgb_return_future = t_trans.target_scaler.inverse_transform(xgb_pred_scaled)[0][0]
                    trans_pred_scaled = transformer_model.predict(X_predict, verbose=0)
                    trans_return_future = t_trans.target_scaler.inverse_transform(trans_pred_scaled)[0][0]
                    last_close = recent_features['close'].iloc[-1]
                    last_date = recent_features.index[-1].strftime('%Y-%m-%d')
                    xgb_val = last_close * (1 + xgb_return_future)
                    trans_val = last_close * (1 + trans_return_future)
                    
                    # Tính toán mức độ rủi ro biến động dựa trên ATR và khoảng giá an toàn
                    last_atr = raw_df['atr_14'].iloc[-1]
                    risk_ratio = (last_atr / last_close) * 100
                    if risk_ratio < 1.5:
                        risk_level = "Thấp (An sau - Thị trường ổn định) 🟢"
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
                        print(f"  - 🌳 XGBoost    : {format_vn(xgb_val)} VNĐ | {xgb_trend} | Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ")
                        print(f"  - 🤖 Transformer: {format_vn(trans_val)} VNĐ | {trans_trend} | Khoảng an toàn: [{format_vn(trans_lower)} - {format_vn(trans_upper)}] VNĐ")
                        log_prediction_to_file(ticker, last_date, last_close, risk_level, risk_ratio, xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper)
                    else:
                        rate_today = usd_vnd_rate
                        print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {format_vn(last_close)} VNĐ (tương đương ${last_close/rate_today:,.2f} USD)")
                        print(f"⚠️ Mức độ rủi ro biến động hiện tại: {risk_level} (Tỷ lệ biến động: {risk_ratio:.2f}%)")
                        print(f"🔮 DỰ BÁO GIÁ MỞ CỬA & KHOẢNG AN TOÀN CHO PHIÊN KẾ TIẾP (tỷ giá quy đổi: 1 USD = {format_vn(rate_today)} VNĐ):")
                        print(f"  - 🌳 XGBoost    : {format_vn(xgb_val)} VNĐ (tương đương ${xgb_val/rate_today:.2f} USD) | {xgb_trend}")
                        print(f"                    Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ (tương đương ${xgb_lower/rate_today:.2f} - ${xgb_upper/rate_today:.2f} USD)")
                        print(f"  - 🤖 Transformer: {format_vn(trans_val)} VNĐ (tương đương ${trans_val/rate_today:.2f} USD) | {trans_trend}")
                        print(f"                    Khoảng an toàn: [{format_vn(trans_lower)} - {format_vn(trans_upper)}] VNĐ (tương đương ${trans_lower/rate_today:.2f} - ${trans_upper/rate_today:.2f} USD)")
                        log_prediction_to_file(ticker, last_date, last_close, risk_level, risk_ratio, xgb_val, xgb_lower, xgb_upper, trans_val, trans_lower, trans_upper, rate_today)
                else:
                    print(f"Lỗi: Không đủ dữ liệu {LOOKBACK_WINDOW} ngày cho {ticker}")
            else:
                print(f"Lỗi: Không tải được dữ liệu trực tuyến cho {ticker}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
