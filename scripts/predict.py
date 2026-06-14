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
        xgb_pred_scaled = xgb_model.predict(X_predict_hybrid)
        xgb_return_future = scaler_y.inverse_transform(xgb_pred_scaled)[0]

        # Dự đoán từ Transformer
        trans_pred_raw = transformer_model.predict(X_predict, verbose=0)
        trans_pred_clean = trans_pred_raw[0] if isinstance(trans_pred_raw, (list, tuple)) else trans_pred_raw
        trans_return_future = scaler_y.inverse_transform(trans_pred_clean)[0]
        
        # Kết quả cuối cùng
        last_close = float(raw_df['close'].iloc[-1])
        last_date = raw_df.index[-1].strftime('%Y-%m-%d')
        
        next_dates = []
        curr_date = pd.to_datetime(last_date)
        for h in [1, 2, 3]:
            curr_date = curr_date + pd.tseries.offsets.BDay(1)
            next_dates.append(curr_date.strftime('%Y-%m-%d'))
            
        xgb_vals = last_close * (1 + xgb_return_future)
        trans_vals = last_close * (1 + trans_return_future)
        
        last_atr = float(raw_df['atr_14'].iloc[-1])
        risk_ratio = (last_atr / last_close) * 100
        
        if risk_ratio < 1.5:
            risk_level = "Thấp (An toàn - Thị trường ổn định) 🟢"
        elif risk_ratio < 3.0:
            risk_level = "Trung bình (Biến động nhẹ - Thận trọng) 🟡"
        else:
            risk_level = "Cao (Nguy hiểm - Biến động cực mạnh) 🔴"
            
        xgb_lowers = xgb_vals - 1.5 * last_atr
        xgb_uppers = xgb_vals + 1.5 * last_atr
        trans_lowers = trans_vals - 1.5 * last_atr
        trans_uppers = trans_vals + 1.5 * last_atr

        # Trích xuất các biến kỹ thuật & vĩ mô phục vụ các Agent
        rsi_14 = float(df_transformed['rsi_14'].iloc[-1])
        macd_ratio = float(df_transformed['macd_ratio'].iloc[-1])
        bb_position = float(df_transformed['bb_position'].iloc[-1])
        mfi_14 = float(df_transformed['mfi_14'].iloc[-1])
        vix_lag1 = float(df_transformed['vix_lag1'].iloc[-1])
        bond_yield_lag1 = float(df_transformed['bond_yield_lag1'].iloc[-1])
        usdvnd_change = float(df_transformed['usdvnd_change'].iloc[-1])
        vnindex_return_lag1 = float(df_transformed['vnindex_return_lag1'].iloc[-1])
        news_sentiment_score = float(df['sentiment_score'].iloc[-1]) if 'sentiment_score' in df.columns else 0.0

        # Import và thực thi các Agent
        from src.agents.technical_agent import TechnicalAgent
        from src.agents.sentiment_agent import SentimentAgent
        from src.agents.macro_agent import MacroAgent
        from src.agents.risk_agent import RiskAgent
        from src.agents.orchestrator import Orchestrator

        tech_agent = TechnicalAgent()
        sent_agent = SentimentAgent()
        macro_agent = MacroAgent()
        risk_agent = RiskAgent()
        orchestrator = Orchestrator()

        tech_rep = tech_agent.analyze(ticker, trans_return_future, xgb_return_future, rsi_14, macd_ratio, bb_position)
        sent_rep = sent_agent.analyze(ticker, news_sentiment_score)
        macro_rep = macro_agent.analyze(ticker, vix_lag1, bond_yield_lag1, usdvnd_change, vnindex_return_lag1)
        risk_rep = risk_agent.analyze(ticker, last_close, last_atr, mfi_14)

        decision = orchestrator.run_debate(ticker, last_close, tech_rep, sent_rep, macro_rep, risk_rep)
        
        print("==========================================================================")
        
        # Ghi nhận kết quả dự đoán vào tệp history log với nhãn mô hình rõ ràng
        logs_dir = os.path.join(ROOT_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "predictions_history.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # In kết quả trực tiếp ra màn hình cho thân thiện
        print(f"📊 KẾT QUẢ DỰ BÁO 3 NGÀY CHO MÃ: {ticker}")
        print(f"  💵 Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ")
        print(f"  ⚠️  Mức độ rủi ro biến động: {risk_level} ({risk_ratio:.2f}%)")
        if "VNM" not in ticker.upper():
            print(f"  💵 Quy đổi USD: ${last_close/USD_TO_VND:,.2f} USD (Tỷ giá: {format_vn(USD_TO_VND)} VNĐ)")
            print("  🌳 Dự báo Hybrid XGBoost (Chuỗi 3 ngày):")
            for h in range(3):
                trend = f"📈 TĂNG ({xgb_return_future[h]*100:+.2f}%)" if xgb_vals[h] >= last_close else f"📉 GIẢM ({xgb_return_future[h]*100:+.2f}%)"
                print(f"     ➔ T+{h+1} ({next_dates[h]}): {format_vn(xgb_vals[h])} VNĐ (${xgb_vals[h]/USD_TO_VND:.2f} USD) | {trend}")
        else:
            print("  🌳 Dự báo Hybrid XGBoost (Chuỗi 3 ngày):")
            for h in range(3):
                trend = f"📈 TĂNG ({xgb_return_future[h]*100:+.2f}%)" if xgb_vals[h] >= last_close else f"📉 GIẢM ({xgb_return_future[h]*100:+.2f}%)"
                print(f"     ➔ T+{h+1} ({next_dates[h]}): {format_vn(xgb_vals[h])} VNĐ | {trend}")

        print("\n==========================================================================")
        print("🤖 BÁO CÁO HỆ THỐNG MULTI-AGENT (PydanticAI & TradingAgents)")
        print("==========================================================================")
        print(tech_rep)
        print(sent_rep)
        print(macro_rep)
        print(risk_rep)
        print("--------------------------------------------------------------------------")
        print("💬 CUỘC TRANH LUẬN BULL VS BEAR:")
        print(decision.debate_summary)
        print("--------------------------------------------------------------------------")
        print("🎯 QUYẾT ĐỊNH ĐẦU TƯ CUỐI CÙNG:")
        print(f"   ➔ Khuyến nghị : {decision.action}")
        print(f"   ➔ Độ tự tin   : {decision.confidence_score*100:.1f}%")
        
        # Tính toán SL/TP tham chiếu và tỷ lệ % tiềm năng lãi/lỗ
        ref_sl = last_close - 2.0 * last_atr
        ref_tp = last_close + 4.0 * last_atr
        profit_pct = (ref_tp - last_close) / last_close * 100
        loss_pct = (ref_sl - last_close) / last_close * 100
        
        if decision.action == "BUY":
            print(f"   ➔ Cắt lỗ (SL)  : {format_vn(decision.stop_loss)} VNĐ ({loss_pct:+.2f}%)")
            print(f"   ➔ Chốt lời (TP): {format_vn(decision.take_profit)} VNĐ ({profit_pct:+.2f}%)")
            print(f"   ➔ Tỷ lệ R/R    : 1:2.0")
        else:
            print(f"   ➔ SL tham chiếu: {format_vn(ref_sl)} VNĐ ({loss_pct:+.2f}%)")
            print(f"   ➔ TP tham chiếu: {format_vn(ref_tp)} VNĐ ({profit_pct:+.2f}%)")
            print(f"   ➔ Tỷ lệ R/R    : 1:2.0")
        print(f"   ➔ Lập luận    : {decision.reasoning}")
        print("==========================================================================")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== BẢN GHI DỰ BÁO 3 NGÀY & MULTI-AGENT REPORT ({timestamp}) ===\n")
            f.write(f"Mã chứng khoán: {ticker}\n")
            if "VNM" in ticker.upper():
                f.write(f"Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ\n")
                f.write(f"Rủi ro biến động: {risk_level} (Tỷ lệ: {risk_ratio:.2f}%)\n")
                f.write("Dự báo Hybrid XGBoost:\n")
                for h in range(3):
                    trend = f"📈 TĂNG ({xgb_return_future[h]*100:+.2f}%)" if xgb_vals[h] >= last_close else f"📉 GIẢM ({xgb_return_future[h]*100:+.2f}%)"
                    f.write(f"  ➔ T+{h+1} ({next_dates[h]}): {format_vn(xgb_vals[h])} VNĐ ({trend}) | Khoảng an toàn: [{format_vn(xgb_lowers[h])} - {format_vn(xgb_uppers[h])}] VNĐ\n")
                f.write("Dự báo Transformer:\n")
                for h in range(3):
                    trend = f"📈 TĂNG ({trans_return_future[h]*100:+.2f}%)" if trans_vals[h] >= last_close else f"📉 GIẢM ({trans_return_future[h]*100:+.2f}%)"
                    f.write(f"  ➔ T+{h+1} ({next_dates[h]}): {format_vn(trans_vals[h])} VNĐ ({trend}) | Khoảng an toàn: [{format_vn(trans_lowers[h])} - {format_vn(trans_uppers[h])}] VNĐ\n")
            else:
                f.write(f"Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ (${last_close/USD_TO_VND:,.2f} USD)\n")
                f.write(f"Rủi ro biến động: {risk_level} (Tỷ lệ: {risk_ratio:.2f}%)\n")
                f.write(f"Tỷ giá USD/VND quy đổi: 1 USD = {format_vn(USD_TO_VND)} VNĐ\n")
                f.write("Dự báo Hybrid XGBoost:\n")
                for h in range(3):
                    trend = f"📈 TĂNG ({xgb_return_future[h]*100:+.2f}%)" if xgb_vals[h] >= last_close else f"📉 GIẢM ({xgb_return_future[h]*100:+.2f}%)"
                    f.write(f"  ➔ T+{h+1} ({next_dates[h]}): {format_vn(xgb_vals[h])} VNĐ (${xgb_vals[h]/USD_TO_VND:.2f} USD) ({trend}) | Khoảng an toàn: [{format_vn(xgb_lowers[h])} - {format_vn(xgb_uppers[h])}] VNĐ\n")
                f.write("Dự báo Transformer:\n")
                for h in range(3):
                    trend = f"📈 TĂNG ({trans_return_future[h]*100:+.2f}%)" if trans_vals[h] >= last_close else f"📉 GIẢM ({trans_return_future[h]*100:+.2f}%)"
                    f.write(f"  ➔ T+{h+1} ({next_dates[h]}): {format_vn(trans_vals[h])} VNĐ (${trans_vals[h]/USD_TO_VND:.2f} USD) ({trend}) | Khoảng an toàn: [{format_vn(trans_lowers[h])} - {format_vn(trans_uppers[h])}] VNĐ\n")
            f.write("\n🤖 BÁO CÁO MULTI-AGENT:\n")
            f.write(f"Khuyến nghị cuối cùng: {decision.action} (Độ tự tin: {decision.confidence_score*100:.1f}%)\n")
            if decision.action == "BUY":
                f.write(f"  * Cắt lỗ (SL)  : {format_vn(decision.stop_loss)} VNĐ ({loss_pct:+.2f}%)\n")
                f.write(f"  * Chốt lời (TP): {format_vn(decision.take_profit)} VNĐ ({profit_pct:+.2f}%)\n")
            else:
                f.write(f"  * SL tham chiếu: {format_vn(ref_sl)} VNĐ ({loss_pct:+.2f}%)\n")
                f.write(f"  * TP tham chiếu: {format_vn(ref_tp)} VNĐ ({profit_pct:+.2f}%)\n")
            f.write(f"Lập luận: {decision.reasoning}\n")
            f.write("-" * 50 + "\n\n")
            
        print("💾 Đã tự động ghi nhận kết quả dự đoán và báo cáo Multi-Agent vào nhật ký lịch sử.")

        
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi tính toán dự đoán: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
