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
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
from src.ai_models import build_transformer

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

def main():
    print("🚀 [HỆ THỐNG] Khởi động Pipeline Huấn luyện RIÊNG mô hình Transformer...")
    
    SENTIMENT_ENGINE = 'finbert'
    TICKERS = ["VNM.VN", "GOOGL", "META"]
    START_TRAIN = "2010-01-01"
    END_PREDICT = "2026-05-20"
    
    # Cấu hình callbacks cho mạng Neural (sử dụng Cosine Decay và Plateau Decay điều phối học tập)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        LearningRateScheduler(cosine_decay, verbose=0)
    ]
    
    models_dir = os.path.join(ROOT_DIR, 'models')
    results_dir = os.path.join(ROOT_DIR, 'results')
    
    for ticker in TICKERS:
        print(f"\n--------------------------------------------------")
        print(f"🔄 BẮT ĐẦU HUẤN LUYỆN TRANSFORMER CHO MÃ: {ticker}...")
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
        
        print(f"📊 Thông tin tập dữ liệu cho {ticker}:")
        print(f"   🔹 Train (Huấn luyện): {len(X_tr)} mẫu")
        print(f"   🔹 Val (Kiểm chứng)  : {len(X_va)} mẫu")
        print(f"   🔹 Test (Kiểm thử)   : {len(X_test_i)} mẫu")
        
        # Load best transformer parameters if they exist
        best_params_path = os.path.join(models_dir, 'best_transformer_params.json')
        d_model = 128
        num_heads = 8
        dropout_rate = 0.3
        learning_rate = 1e-4
        batch_size = 64
        
        if os.path.exists(best_params_path):
            try:
                import json
                with open(best_params_path, 'r') as f:
                    best_params = json.load(f)
                d_model = best_params.get('d_model', d_model)
                num_heads = best_params.get('num_heads', num_heads)
                dropout_rate = best_params.get('dropout_rate', dropout_rate)
                learning_rate = best_params.get('learning_rate', learning_rate)
                batch_size = best_params.get('batch_size', batch_size)
                print(f"🥇 [TUNED] Đang sử dụng siêu tham số tối ưu từ Optuna:")
                print(f"   - d_model: {d_model}, num_heads: {num_heads}, dropout: {dropout_rate:.4f}, lr: {learning_rate:.6f}, batch: {batch_size}")
            except Exception as e:
                print(f"⚠️ Không thể đọc best_params_path: {e}")
                
        # Huấn luyện Transformer (In thông tin chi tiết qua verbose=2 để người dùng theo dõi loss)
        print(f"🤖 [TRAIN] Đang huấn luyện Transformer...")
        transformer_model = build_transformer(
            input_shape=(X_tr.shape[1], X_tr.shape[2]),
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        history = transformer_model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2
        )

        
        # Đánh giá kết quả trên tập kiểm thử (Test set)
        split_idx = int(len(X_3D) * 0.8)
        df_align = df.iloc[transformer.time_steps:].reset_index(drop=True)
        test_close_i = df_align.loc[split_idx:, 'close'].values[:len(X_test_i)]
        y_test_true_i = test_close_i * (1 + y_test_raw_i)
        
        # Dự báo Transformer
        trans_scaled = transformer_model.predict(X_test_i, verbose=0)
        trans_return = transformer.target_scaler.inverse_transform(trans_scaled).ravel()
        trans_preds = test_close_i * (1 + trans_return)
        
        print(f"\n📊 BẢNG ĐÁNH GIÁ SAI SỐ CHO MÃ: {ticker}")
        evaluate_predictions(y_test_true_i, trans_preds, f"Transformer ({ticker})", ticker)
        
        # --- Trực quan hóa kết quả ---
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_true_i[-100:], label="Giá thực tế (VNĐ)", color='black', linewidth=2.5)
        plt.plot(trans_preds[-100:], label="Dự báo Transformer (VNĐ)", color='blue', linestyle='-.')
        plt.ylabel("Giá (VNĐ)", fontsize=12)
        plt.title(f"XU HƯỚNG GIÁ THỰC TẾ VS DỰ BÁO CỦA {ticker} (MÔ HÌNH TRANSFORMER)", fontsize=14, fontweight='bold')
        plt.xlabel("100 Phiên giao dịch cuối cùng (Tập Test)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'transformer_result_{ticker}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 Đã xuất biểu đồ cho {ticker} vào file: '{plot_path}'")
        
        # Vẽ biểu đồ suy giảm Loss (Huấn luyện vs Kiểm chứng)
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Loss Huấn luyện')
        plt.plot(history.history['val_loss'], label='Loss Kiểm chứng')
        plt.title(f"BIỂU ĐỒ SUY GIẢM LOSS CỦA {ticker}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Huber)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(results_dir, f'transformer_loss_{ticker}.png')
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        print(f"💾 Đã xuất biểu đồ Loss cho {ticker} vào file: '{loss_plot_path}'")
        
        # Lưu trữ các file mô hình
        os.makedirs(models_dir, exist_ok=True)
        transformer_model.save(os.path.join(models_dir, f'transformer_model_{ticker}.keras'))
        joblib.dump(transformer.feature_scaler, os.path.join(models_dir, f'feature_scaler_{ticker}.pkl'))
        joblib.dump(transformer.target_scaler, os.path.join(models_dir, f'target_scaler_{ticker}.pkl'))
        print(f"💾 Đã lưu model Transformer & Scalers cho {ticker}.")

if __name__ == "__main__":
    main()
