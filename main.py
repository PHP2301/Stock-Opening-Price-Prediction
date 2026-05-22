import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import joblib
import yfinance as yf
import pandas_ta as ta

# ==========================================
# CỐ ĐỊNH RANDOM SEED — ĐẢM BẢO KẾT QUẢ TÁI LẬP
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from src.data_loader import fetch_and_prepare_data, format_vn
from src.features import DataTransformer
from src.ai_models import build_xgboost_optimized, build_transformer

USD_TO_VND = 25400
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

def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("🚀 [HỆ THỐNG] Khởi động Pipeline Huấn luyện (Môi trường Đa mã Quy đổi VNĐ)...")
    
    TICKERS = ["VNM.VN", "GOOGL", "META"]
    START_TRAIN = "2010-01-01"
    END_PREDICT = "2026-05-20"
    
    # 💡 CẤU HÌNH QUAN TRỌNG: Đặt INDIVIDUAL_TRAINING = True để huấn luyện riêng cho từng mã.
    # Huấn luyện riêng biệt giúp mô hình học chính xác đặc thù của từng loại cổ phiếu, giảm sai số đáng kể.
    INDIVIDUAL_TRAINING = True 
    
    # Early stopping cấu hình thông minh cho mạng Neural
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
    ]
    
    if INDIVIDUAL_TRAINING:
        print("\n" + "="*60)
        print("🎯 PHƯƠNG ÁN 1: HUẤN LUYỆN ĐỘC LẬP CHO TỪNG CỔ PHIẾU (TỐI ƯU SAI SỐ)")
        print("="*60 + "\n")
        
        for ticker in TICKERS:
            print(f"\n--------------------------------------------------")
            print(f"🔄 BẮT ĐẦU HUẤN LUYỆN RIÊNG CHO MÃ: {ticker}...")
            print(f"--------------------------------------------------")
            
            df = fetch_and_prepare_data(ticker, start_date=START_TRAIN, end_date=END_PREDICT)
            
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
            
            os.makedirs('results', exist_ok=True)
            plot_path = os.path.join('results', f'model_battle_result_{ticker}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"💾 Đã xuất biểu đồ cho {ticker} vào file: '{plot_path}'")
            
            # Lưu mô hình độc lập cho mã này
            os.makedirs('models', exist_ok=True)
            joblib.dump(xgb_model, f'models/xgboost_model_{ticker}.pkl')
            transformer_model.save(f'models/transformer_model_{ticker}.keras')
            joblib.dump(transformer.feature_scaler, f'models/feature_scaler_{ticker}.pkl')
            joblib.dump(transformer.target_scaler, f'models/target_scaler_{ticker}.pkl')
            
            # --- DỰ BÁO CHO PHIÊN KẾ TIẾP CỦA MÃ NÀY ---
            print(f"\n🔮 BƯỚC 8: DỰ BÁO GIÁ MỞ CỬA KẾ TIẾP CHO {ticker}...")
            raw_df = yf.download(ticker, period="150d", progress=False)
            if not raw_df.empty:
                if type(raw_df.columns) == pd.MultiIndex:
                    raw_df.columns = raw_df.columns.droplevel(1)
                raw_df.columns = [col.lower() for col in raw_df.columns]
                
                # Quy đổi giá trị từ USD sang VNĐ cho GOOGL/META trước khi tính các chỉ báo
                if "VNM" not in ticker.upper():
                    price_cols = ['open', 'high', 'low', 'close']
                    for col in price_cols:
                        raw_df[col] = raw_df[col] * USD_TO_VND
                
                # Tính các chỉ báo
                raw_df['rsi_14'] = ta.rsi(raw_df['close'], length=14)
                macd_df = ta.macd(raw_df['close'], fast=12, slow=26, signal=9)
                raw_df = pd.concat([raw_df, macd_df], axis=1)
                raw_df['volatility_20'] = raw_df['close'].pct_change().rolling(window=20).std()
                raw_df['close_lag1'] = raw_df['close'].shift(1)
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
                    
                    if "VNM" in ticker.upper():
                        print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {format_vn(last_close)} VNĐ")
                        print(f"🔮 DỰ BÁO GIÁ MỞ CỬA CHO PHIÊN KẾ TIẾP:")
                        print(f"  - 🌳 XGBoost    : {format_vn(xgb_val)} VNĐ")
                        print(f"  - 🤖 Transformer: {format_vn(trans_val)} VNĐ")
                    else:
                        print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {format_vn(last_close)} VNĐ (tương đương ${last_close/USD_TO_VND:,.2f} USD)")
                        print(f"🔮 DỰ BÁO GIÁ MỞ CỬA CHO PHIÊN KẾ TIẾP:")
                        print(f"  - 🌳 XGBoost    : {format_vn(xgb_val)} VNĐ (tương đương ${xgb_val/USD_TO_VND:.2f} USD)")
                        print(f"  - 🤖 Transformer: {format_vn(trans_val)} VNĐ (tương đương ${trans_val/USD_TO_VND:.2f} USD)")
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
            df = fetch_and_prepare_data(ticker, start_date=START_TRAIN, end_date=END_PREDICT)
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
        
        # 5.1 Huấn luyện XGBoost
        print("🧠 [TRAIN] Đang huấn luyện mô hình XGBoost chung...")
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        xgb_model = build_xgboost_optimized(X_train_flat, y_train)
        
        # 5.2 Huấn luyện Transformer
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
            os.makedirs('results', exist_ok=True)
            plot_path = os.path.join('results', f'model_battle_result_{ticker}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            os.makedirs('models', exist_ok=True)
            joblib.dump(xgb_model, 'models/xgboost_model.pkl')
            transformer_model.save('models/transformer_model.keras')
            for ticker in TICKERS:
                joblib.dump(transformers[ticker].feature_scaler, f'models/feature_scaler_{ticker}.pkl')
                joblib.dump(transformers[ticker].target_scaler, f'models/target_scaler_{ticker}.pkl')

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
                    if "VNM" not in ticker.upper():
                        price_cols = ['open', 'high', 'low', 'close']
                        for col in price_cols:
                            raw_df[col] = raw_df[col] * USD_TO_VND
                    raw_df['rsi_14'] = ta.rsi(raw_df['close'], length=14)
                    macd_df = ta.macd(raw_df['close'], fast=12, slow=26, signal=9)
                    raw_df = pd.concat([raw_df, macd_df], axis=1)
                    raw_df['volatility_20'] = raw_df['close'].pct_change().rolling(window=20).std()
                    raw_df['close_lag1'] = raw_df['close'].shift(1)
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
                        
                        if "VNM" in ticker.upper():
                            print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {format_vn(last_close)} VNĐ")
                            print(f"🔮 DỰ BÁO GIÁ MỞ CỬA CHO PHIÊN KẾ TIẾP:")
                            print(f"  - 🌳 XGBoost    : {format_vn(xgb_val)} VNĐ")
                            print(f"  - 🤖 Transformer: {format_vn(trans_val)} VNĐ")
                        else:
                            print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {format_vn(last_close)} VNĐ (tương đương ${last_close/USD_TO_VND:,.2f} USD)")
                            print(f"🔮 DỰ BÁO GIÁ MỞ CỬA CHO PHIÊN KẾ TIẾP:")
                            print(f"  - 🌳 XGBoost    : {format_vn(xgb_val)} VNĐ (tương đương ${xgb_val/USD_TO_VND:.2f} USD)")
                            print(f"  - 🤖 Transformer: {format_vn(trans_val)} VNĐ (tương đương ${trans_val/USD_TO_VND:.2f} USD)")
                    else:
                        print(f"Lỗi: Không đủ dữ liệu {LOOKBACK_WINDOW} ngày cho {ticker}")
                else:
                    print(f"Lỗi: Không tải được dữ liệu trực tuyến cho {ticker}")
            print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()