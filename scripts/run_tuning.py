import os
import random
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import optuna

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thiết lập đường dẫn thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data
from src.features import DataTransformer
from src.ai_models import build_transformer

# Cố định random seed để kết quả tái lập tốt nhất
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Biến toàn cục chứa dữ liệu phục vụ cho hàm mục tiêu Optuna
X_train = None
y_train = None
X_val = None
y_val = None

def objective(trial):
    # Tránh in quá nhiều logs của TF trong các trial
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # 1. Gợi ý các siêu tham số từ không gian tìm kiếm
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    
    # 2. Xây dựng mô hình với tham số thử nghiệm
    model = build_transformer(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        d_model=d_model,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Early stopping để dừng sớm các cấu hình tệ
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)
    
    # Huấn luyện nhanh mô hình (epochs thấp để tối ưu thời gian quét)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=35,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Lấy val_loss nhỏ nhất đạt được làm thước đo
    val_loss = min(history.history['val_loss'])
    return val_loss

if __name__ == "__main__":
    import sys
    
    # Danh sách tickers mặc định
    TICKERS = ["VNM.VN", "GOOGL", "META"]
    
    # Cho phép chọn ticker cụ thể qua command line
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
            print(f"🎯 Chỉ chạy Tuning Optuna cho mã: {TICKERS[0]}")
        elif arg == "ALL":
            print(f"🎯 Chạy Tuning Optuna cho toàn bộ watchlist: {TICKERS}")
    else:
        print(f"🎯 Chạy Tuning Optuna tuần tự cho toàn bộ watchlist: {TICKERS}")
        
    for t_ticker in TICKERS:
        print(f"\n==================================================")
        print(f"🚀 [OPTUNA] Khởi động Tìm kiếm Siêu tham số cho: {t_ticker}")
        print(f"==================================================")
        
        # Tải và chuẩn bị dữ liệu
        df = fetch_and_prepare_data(t_ticker, start_date="2012-01-01", end_date="2026-05-20")
        
        # Tạo đặc trưng dừng
        transformer = DataTransformer(time_steps=45)
        X_scaled, y_scaled = transformer.fit_transform_data(df)
        X_3D, y_3D = transformer.create_sliding_windows(X_scaled, y_scaled)
        
        # Tách 80/20 train/test
        X_train_all, y_train_all, _, _, _ = transformer.split_train_test_chronological(df, X_3D, y_3D, train_ratio=0.8)
        
        # Tách 90/10 train/val phục vụ cho Optuna
        val_size = int(len(X_train_all) * 0.1)
        
        # Gán biến toàn cục
        X_train = X_train_all[:-val_size]
        y_train = y_train_all[:-val_size]
        X_val = X_train_all[-val_size:]
        y_val = y_train_all[-val_size:]
        
        print(f"📊 Dữ liệu tuning {t_ticker}: Train = {X_train.shape[0]} mẫu, Val = {X_val.shape[0]} mẫu")
        print(f"⏳ Bắt đầu quét thử nghiệm 15 cấu hình khác nhau...")
        
        # Tắt thông báo rườm rà của Optuna để hiển thị gọn gàng hơn
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15)
        
        print(f"\n🏆 KẾT QUẢ TỐI ƯU HÓA HOÀN TẤT CHO {t_ticker}!")
        print(f"🥇 Cấu hình tốt nhất đạt Val Loss = {study.best_value:.6f}")
        best_params = study.best_params
        
        for key, value in best_params.items():
            print(f"   🔹 {key}: {value}")
            
        # Lưu các tham số tối ưu vào thư mục config/ để dùng riêng cho ticker đó
        config_dir = os.path.join(ROOT_DIR, 'config')
        os.makedirs(config_dir, exist_ok=True)
        best_params_path = os.path.join(config_dir, f'best_transformer_params_{t_ticker}.json')
        
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
            
        print(f"💾 Đã lưu cấu hình tốt nhất vào file: '{best_params_path}'")
