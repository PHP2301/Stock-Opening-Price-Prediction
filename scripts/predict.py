import os
import sys
import datetime
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import pandas_ta as ta

# Cố định random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thiết lập đường dẫn thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data, format_vn, get_realtime_usd_vnd_rate
from src.features import DataTransformer
from src.ai_models import PositionalEmbedding, TimeDecayAttention, MultiTaskModel, UncertaintyWeightsLayer

USD_TO_VND = get_realtime_usd_vnd_rate()
LOOKBACK_WINDOW = 45

def main():
    ticker = "VNM.VN"
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        
    print(f"🔮 BẮT ĐẦU CHẠY DỰ BÁO GIÁ MỞ CỬA CHO MÃ: {ticker}")
    print(f"💵 Tỷ giá USD/VND hiện tại: {format_vn(USD_TO_VND)} VNĐ\n")
    
    models_dir = os.path.join(ROOT_DIR, "models")
    xgb_path = os.path.join(models_dir, f"xgboost_model_{ticker}.pkl")
    trans_path = os.path.join(models_dir, f"transformer_model_{ticker}.keras")
    scaler_x_path = os.path.join(models_dir, f"feature_scaler_{ticker}.pkl")
    scaler_y_path = os.path.join(models_dir, f"target_scaler_{ticker}.pkl")
    
    # Kiểm tra sự tồn tại của mô hình
    if not (os.path.exists(xgb_path) and os.path.exists(trans_path) and os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path)):
        print(f"❌ LỖI: Không tìm thấy đầy đủ mô hình đã huấn luyện cho mã {ticker}!")
        print(f"Vui lòng chạy lệnh huấn luyện trước: python scripts/run_training.py {ticker}")
        sys.exit(1)
        
    print("⏳ Đang nạp mô hình và bộ chuẩn hóa...")
    try:
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        xgb_model = joblib.load(xgb_path)
        
        # Load model Keras 3 Functional
        transformer_model = tf.keras.models.load_model(
            trans_path, 
            custom_objects={
                'PositionalEmbedding': PositionalEmbedding,
                'TimeDecayAttention': TimeDecayAttention,
                'MultiTaskModel': MultiTaskModel,
                'UncertaintyWeightsLayer': UncertaintyWeightsLayer
            },
            safe_mode=False
        )
        print("✅ Nạp mô hình thành công.")
    except Exception as e:
        print(f"❌ Lỗi nạp mô hình: {e}")
        sys.exit(1)
        
    print(f"⏳ Đang tải và chuẩn bị dữ liệu mới nhất cho {ticker}...")
    try:
        # Tải dữ liệu để tạo đặc trưng
        start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
        end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        df = fetch_and_prepare_data(ticker, start_date=start_date, end_date=end_date, sentiment_engine="vader")
        if df.empty:
            print("❌ Lỗi: Không thể tải dữ liệu giao dịch từ Yahoo Finance / DNSE.")
            sys.exit(1)
            
        df = df.sort_values('date').reset_index(drop=True)
        # Tính toán chỉ báo ATR 14 để đo lường khoảng an toàn
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        transformer = DataTransformer(time_steps=LOOKBACK_WINDOW)
        # SỬA: Phải gọi transform_df() để sinh ra 34 features trước khi lọc cột đặc trưng
        df_transformed = transformer.transform_df(df)
        recent_features = df_transformed[transformer.feature_cols].tail(LOOKBACK_WINDOW)
        
        if len(recent_features) < LOOKBACK_WINDOW:
            print(f"❌ Lỗi: Không đủ dữ liệu {LOOKBACK_WINDOW} ngày để tạo sliding window.")
            sys.exit(1)
            
        raw_df = df.tail(LOOKBACK_WINDOW).copy()
        raw_df.set_index('date', inplace=True)
        
        # Scale inputs
        recent_scaled = scaler_X.transform(recent_features.values)
        X_predict = recent_scaled.reshape(1, LOOKBACK_WINDOW, len(transformer.feature_cols))
        
        # Chạy dummy forward pass để khởi tạo các thuộc tính Functional của Keras 3
        _ = transformer_model(X_predict)
        
        # Thiết lập bộ trích xuất đặc trưng ẩn
        feature_extractor = tf.keras.models.Model(
            inputs=transformer_model.input,
            outputs=transformer_model.get_layer("latent_embedding").output
        )
        
        # Dự đoán
        X_predict_latent = feature_extractor.predict(X_predict, verbose=0)
        X_predict_today = X_predict[0, -1, :].reshape(1, -1)
        X_predict_hybrid = np.concatenate([X_predict_latent, X_predict_today], axis=1)
        
        # Dự đoán từ XGBoost Hybrid
        xgb_pred_scaled = xgb_model.predict(X_predict_hybrid).reshape(-1, 1)
        xgb_return_future = scaler_y.inverse_transform(xgb_pred_scaled)[0][0]
        
        # Kết quả cuối cùng
        last_close = float(raw_df['close'].iloc[-1])
        last_date = raw_df.index[-1].strftime('%Y-%m-%d')
        
        xgb_val = last_close * (1 + xgb_return_future)
        
        last_atr = float(raw_df['atr_14'].iloc[-1])
        risk_ratio = (last_atr / last_close) * 100
        
        if risk_ratio < 1.5:
            risk_level = "Thấp (An toàn - Thị trường ổn định) 🟢"
        elif risk_ratio < 3.0:
            risk_level = "Trung bình (Biến động nhẹ - Thận trọng) 🟡"
        else:
            risk_level = "Cao (Nguy hiểm - Biến động cực mạnh) 🔴"
            
        xgb_lower = xgb_val - 1.5 * last_atr
        xgb_upper = xgb_val + 1.5 * last_atr
        
        xgb_trend = f"📈 TĂNG ({((xgb_val - last_close)/last_close)*100:+.2f}%)" if xgb_val >= last_close else f"📉 GIẢM ({((xgb_val - last_close)/last_close)*100:+.2f}%)"
        
        print("==========================================================================")
        
        # Ghi nhận kết quả dự đoán vào tệp history log với nhãn mô hình rõ ràng
        logs_dir = os.path.join(ROOT_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "predictions_history.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # In kết quả trực tiếp ra màn hình cho thân thiện
        print(f"📊 KẾT QUẢ DỰ BÁO T+3 CHO MÃ: {ticker}")
        print(f"  💵 Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ")
        print(f"  ⚠️  Mức độ rủi ro biến động: {risk_level} ({risk_ratio:.2f}%)")
        if "VNM" not in ticker.upper():
            print(f"  💵 Quy đổi USD: ${last_close/USD_TO_VND:,.2f} USD (Tỷ giá: {format_vn(USD_TO_VND)} VNĐ)")
            print(f"  🌳 Dự báo Hybrid XGBoost (giá đóng cửa sau 3 phiên - T+3): {format_vn(xgb_val)} VNĐ (${xgb_val/USD_TO_VND:.2f} USD) | {xgb_trend}")
        else:
            print(f"  🌳 Dự báo Hybrid XGBoost (giá đóng cửa sau 3 phiên - T+3): {format_vn(xgb_val)} VNĐ | {xgb_trend}")
        print("==========================================================================")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== BẢN GHI DỰ BÁO T+3 ({timestamp}) ===\n")
            f.write(f"Mã chứng khoán: {ticker}\n")
            if "VNM" in ticker.upper():
                f.write(f"Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ\n")
                f.write(f"Rủi ro biến động: {risk_level} (Tỷ lệ: {risk_ratio:.2f}%)\n")
                f.write(f"Dự báo Hybrid XGBoost (giá đóng cửa sau 3 phiên - T+3): {format_vn(xgb_val)} VNĐ ({xgb_trend}) | Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ\n")
            else:
                f.write(f"Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ (${last_close/USD_TO_VND:,.2f} USD)\n")
                f.write(f"Rủi ro biến động: {risk_level} (Tỷ lệ: {risk_ratio:.2f}%)\n")
                f.write(f"Tỷ giá USD/VND quy đổi: 1 USD = {format_vn(USD_TO_VND)} VNĐ\n")
                f.write(f"Dự báo Hybrid XGBoost (giá đóng cửa sau 3 phiên - T+3): {format_vn(xgb_val)} VNĐ (${xgb_val/USD_TO_VND:.2f} USD) ({xgb_trend}) | Khoảng an toàn: [{format_vn(xgb_lower)} - {format_vn(xgb_upper)}] VNĐ\n")
            f.write("-" * 50 + "\n\n")
            
        print("💾 Đã tự động ghi nhận kết quả dự đoán vào nhật ký lịch sử.")

        
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi tính toán dự đoán: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
