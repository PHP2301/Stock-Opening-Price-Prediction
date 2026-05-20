import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import các hàm và class từ các Module bạn đã viết trong thư mục src
from src.data_loader import fetch_and_prepare_data
from src.features import DataTransformer
from src.ai_models import build_xgboost_optimized, build_lstm, build_transformer

def evaluate_predictions(y_true, y_pred, model_name):
    """Thuật toán đánh giá sai số mô hình bằng giá trị tiền tệ thực tế (VNĐ)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"=== KẾT QUẢ MÔ HÌNH {model_name.upper()} ===")
    print(f"❌ Sai số RMSE: {rmse:.4f}")
    print(f"🎯 Sai số MAE : {mae:.4f} (Lệch khoảng {mae:,.0f} VNĐ)\n")
    return rmse, mae

def main():
    print("🚀 [HỆ THỐNG] Khởi động Pipeline Dự báo Giá mở cửa nâng cao...")
    
    # Mốc thời gian thiết lập hệ thống
    TICKER = "VNM.VN"
    START_TRAIN = "2015-01-01"
    END_PREDICT = "2026-05-20" 

    # ==========================================
    # BƯỚC 1 & 2: Tải dữ liệu & Thêm chỉ báo kỹ thuật
    # ==========================================
    df = fetch_and_prepare_data(TICKER, start_date=START_TRAIN, end_date=END_PREDICT)

    # ==========================================
    # BƯỚC 3 & 4: Feature Engineering (Scaling, Tạo mảng 3D, Chia tập Train/Test)
    # ==========================================
    transformer = DataTransformer(time_steps=30)
    X_scaled, y_scaled = transformer.fit_transform_data(df)
    X_3D, y_3D = transformer.create_sliding_windows(X_scaled, y_scaled)
    
    X_train, y_train, X_test, y_test, y_test_raw = transformer.split_train_test_by_year(df, X_3D, y_3D)

    # 🛠️ CHUẨN BỊ DỮ LIỆU ĐỂ DỊCH NGƯỢC GIÁ THỰC TẾ
    # Lấy giá đóng cửa ngày hôm nay (close) ứng với tập Test (2024-2025) để phục vụ công thức nhân ngược
    df_align = df.iloc[transformer.time_steps:].reset_index(drop=True)
    df_align['date'] = pd.to_datetime(df_align['date'])
    test_mask = df_align['date'].dt.year >= 2024
    test_close_prices = df_align.loc[test_mask, 'close'].values[:len(X_test)]
    
    # Ép ngược y_test_raw (đang là tỷ suất lợi nhuận) về giá mở cửa thực tế của ngày mai
    y_test_true_prices = test_close_prices * (1 + y_test_raw)

    # Cấu hình bộ điều khiển huấn luyện thông minh cho Deep Learning
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]

    # ==========================================
    # BƯỚC 5: HUẤN LUYỆN BỘ BA MÔ HÌNH AI
    # ==========================================
    
    # --- 5.1 Huấn luyện XGBoost ---
    print("\n🧠 [TRAIN] Đang huấn luyện mô hình XGBoost + GridSearchCV...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    xgb_model = build_xgboost_optimized(X_train_flat, y_train)
    
    # Dự đoán tỷ suất lợi nhuận và dịch ngược scaling
    xgb_preds_scaled = xgb_model.predict(X_test_flat).reshape(-1, 1)
    xgb_return_preds = transformer.target_scaler.inverse_transform(xgb_preds_scaled).ravel()
    
    # 🛠️ Ép ngược về giá tiền thực tế: Open_dự_báo = Close_hôm_nay * (1 + Return_dự_báo)
    xgb_preds = test_close_prices * (1 + xgb_return_preds)

    # --- 5.2 Huấn luyện LSTM ---
    print("\n🧠 [TRAIN] Đang huấn luyện mạng Deep Learning LSTM (Tối ưu chu kỳ)...")
    lstm_model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Tăng epochs lên 100, truyền validation_data và callbacks để tối ưu độ chính xác
    lstm_model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test),
        epochs=100, 
        batch_size=32, 
        callbacks=callbacks,
        verbose=1 # Hiển thị tiến trình học để theo dõi
    )
    
    # Dự đoán tỷ suất và dịch ngược về giá thực tế
    lstm_preds_scaled = lstm_model.predict(X_test, verbose=0)
    lstm_return_preds = transformer.target_scaler.inverse_transform(lstm_preds_scaled).ravel()
    lstm_preds = test_close_prices * (1 + lstm_return_preds)

    # --- 5.3 Huấn luyện Transformer ---
    print("\n🧠 [TRAIN] Đang huấn luyện mạng Deep Learning Transformer (Tối ưu chu kỳ)...")
    transformer_model = build_transformer(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Đồng bộ tối ưu 100 epochs với Callbacks cho Transformer
    transformer_model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test),
        epochs=100, 
        batch_size=32, 
        callbacks=callbacks,
        verbose=1
    )
    
    # Dự đoán tỷ suất và dịch ngược về giá thực tế
    trans_preds_scaled = transformer_model.predict(X_test, verbose=0)
    trans_return_preds = transformer.target_scaler.inverse_transform(trans_preds_scaled).ravel()
    trans_preds = test_close_prices * (1 + trans_return_preds)

    # ==========================================
    # BƯỚC 6: ĐÁNH GIÁ SAI SỐ THỰC TẾ (2024 - 2025)
    # ==========================================
    print("\n" + "="*50)
    print("📊 BẢNG SO SÁNH SAI SỐ TRÊN TẬP KIỂM THỬ ĐÃ CHUẨN HÓA (VNĐ)")
    print("="*50)
    evaluate_predictions(y_test_true_prices, xgb_preds, "XGBoost")
    evaluate_predictions(y_test_true_prices, lstm_preds, "LSTM")
    evaluate_predictions(y_test_true_prices, trans_preds, "Transformer")

    # ==========================================
    # BƯỚC 7: TRỰC QUAN HÓA KẾT QUẢ KHOA HỌC
    # ==========================================
    plt.figure(figsize=(15, 7))
    # Trực quan hóa 100 phiên giao dịch cuối cùng để phân tích rõ nét cấu trúc sóng giá
    plt.plot(y_test_true_prices[-100:], label="Giá thực tế (Actual Price)", color='black', linewidth=2.5)
    plt.plot(xgb_preds[-100:], label="Dự báo của XGBoost", color='red', linestyle='--')
    plt.plot(lstm_preds[-100:], label="Dự báo của LSTM", color='blue', linestyle='-.')
    plt.plot(trans_preds[-100:], label="Dự báo của Transformer", color='green', linestyle=':')
    
    plt.title(f"XU HƯỚNG GIÁ THỰC TẾ VS ĐẦU RA DỰ BÁO QUA TỶ SUẤT LỢI NHUẬN ({TICKER})", fontsize=14, fontweight='bold')
    plt.xlabel("100 Phiên giao dịch cuối cùng (Tập Test)", fontsize=12)
    plt.ylabel("Giá mở cửa (VNĐ)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Lưu kết quả
    plt.savefig('model_battle_result.png', dpi=300)
    print("💾 Đã xuất biểu đồ so sánh khoa học vào file: 'model_battle_result.png'")
    # ==========================================
    # BƯỚC 8: DỰ BÁO GIÁ MỞ CỬA PHIÊN TIẾP THEO (TƯƠNG LAI)
    # ==========================================
    print("\n" + "="*50)
    print("🚀 BƯỚC 8: DỰ BÁO GIÁ MỞ CỬA CHO PHIÊN GIAO DỊCH KẾ TIẾP (NGÀY MAI)")
    print("="*50)
    
    import yfinance as yf
    import pandas_ta as ta
    
    print(f"Đang tải dữ liệu thời gian thực mới nhất cho {TICKER}...")
    raw_df = yf.download(TICKER, period="60d", progress=False)
    if not raw_df.empty:
        if type(raw_df.columns) == pd.MultiIndex:
            raw_df.columns = raw_df.columns.droplevel(1)
        raw_df.columns = [col.lower() for col in raw_df.columns]
        
        # Tính toán lại bộ đặc trưng
        raw_df['rsi_14'] = ta.rsi(raw_df['close'], length=14)
        macd_df = ta.macd(raw_df['close'], fast=12, slow=26, signal=9)
        raw_df = pd.concat([raw_df, macd_df], axis=1)
        raw_df['volatility_20'] = raw_df['close'].pct_change().rolling(window=20).std()
        raw_df['close_lag1'] = raw_df['close'].shift(1)
        raw_df['volume_change'] = raw_df['volume'].pct_change()
        raw_df['intraday_return'] = (raw_df['close'] - raw_df['open']) / raw_df['open']
        
        # Trích xuất 30 ngày gần nhất làm đầu vào
        recent_features = raw_df[['close', 'rsi_14', 'MACD_12_26_9', 'volatility_20', 'close_lag1', 'volume_change', 'intraday_return']].dropna().tail(30)
        
        if len(recent_features) == 30:
            # Chuẩn hóa đầu vào
            recent_scaled = transformer.feature_scaler.transform(recent_features.values)
            X_predict = recent_scaled.reshape(1, 30, 7)
            
            # Dự báo với XGBoost (chỉ cần reshape về 2D cho scikit-learn)
            xgb_pred_scaled = xgb_model.predict(X_predict.reshape(1, -1)).reshape(-1, 1)
            xgb_return_future = transformer.target_scaler.inverse_transform(xgb_pred_scaled)[0][0]
            
            # Dự báo với LSTM & Transformer
            lstm_pred_scaled = lstm_model.predict(X_predict, verbose=0)
            lstm_return_future = transformer.target_scaler.inverse_transform(lstm_pred_scaled)[0][0]
            
            trans_pred_scaled = transformer_model.predict(X_predict, verbose=0)
            trans_return_future = transformer.target_scaler.inverse_transform(trans_pred_scaled)[0][0]
            
            # Giá đóng cửa hiện tại
            last_close = recent_features['close'].iloc[-1]
            last_date = recent_features.index[-1].strftime('%Y-%m-%d')
            
            print(f"💵 Giá đóng cửa thực tế gần nhất ({last_date}): {last_close:,.0f} VNĐ")
            print("\n🔮 DỰ BÁO GIÁ MỞ CỬA (OPEN) CHO NGÀY GIAO DỊCH TIẾP THEO:")
            print(f"  - 🌳 XGBoost    : {(last_close * (1 + xgb_return_future)):,.0f} VNĐ")
            print(f"  - 🧠 LSTM       : {(last_close * (1 + lstm_return_future)):,.0f} VNĐ")
            print(f"  - 🤖 Transformer: {(last_close * (1 + trans_return_future)):,.0f} VNĐ")
            print("="*50 + "\n")
        else:
            print("Lỗi: Không đủ dữ liệu 30 ngày sau khi làm sạch để dự báo tương lai.")
            
    plt.show()

if __name__ == "__main__":
    main()