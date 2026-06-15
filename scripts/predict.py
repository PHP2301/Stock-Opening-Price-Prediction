import os
import sys
import datetime
import joblib
import json
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
from src.ai_models import PositionalEmbedding, TimeDecayAttention, MultiTaskModel, UncertaintyWeightsLayer, CustomLambda

USD_TO_VND = get_realtime_usd_vnd_rate()
LOOKBACK_WINDOW = 45

def run_prediction_for_ticker(ticker):
    print(f"\n🔮 BẮT ĐẦU CHẠY DỰ BÁO GIÁ MỞ CỬA CHO MÃ: {ticker}")
    print(f"💵 Tỷ giá USD/VND hiện tại: {format_vn(USD_TO_VND)} VNĐ\n")
    
    models_dir = os.path.join(ROOT_DIR, "models")
    trans_path = os.path.join(models_dir, f"transformer_model_{ticker}.keras")
    xgb_path = os.path.join(models_dir, f"xgboost_model_{ticker}.pkl")
    scaler_x_path = os.path.join(models_dir, f"feature_scaler_{ticker}.pkl")
    scaler_y_path = os.path.join(models_dir, f"target_scaler_{ticker}.pkl")
    
    # Kiểm tra sự tồn tại của mô hình
    if not (os.path.exists(trans_path) and os.path.exists(xgb_path) and os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path)):
        print(f"❌ LỖI: Không tìm thấy đầy đủ mô hình đã huấn luyện (Transformer + XGBoost) cho mã {ticker}!")
        print(f"Vui lòng chạy lệnh huấn luyện trước: python scripts/run_training.py {ticker}")
        return
        
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
                'UncertaintyWeightsLayer': UncertaintyWeightsLayer,
                'Lambda': CustomLambda
            },
            safe_mode=False
        )
        print("✅ Nạp mô hình thành công.")
    except Exception as e:
        print(f"❌ Lỗi nạp mô hình cho {ticker}: {e}")
        return
        
    print(f"⏳ Đang tải và chuẩn bị dữ liệu mới nhất cho {ticker}...")
    try:
        # Tải dữ liệu để tạo đặc trưng
        start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
        end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        df = fetch_and_prepare_data(ticker, start_date=start_date, end_date=end_date, sentiment_engine="vader")
        if df.empty:
            print(f"❌ Lỗi: Không thể tải dữ liệu giao dịch cho {ticker} từ Yahoo Finance / DNSE.")
            return
            
        df = df.sort_values('date').reset_index(drop=True)
        # Tính toán chỉ báo ATR 14 để đo lường khoảng an toàn
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        

        transformer = DataTransformer(time_steps=LOOKBACK_WINDOW)
        # SỬA: Phải gọi transform_df() để sinh ra 34 features trước khi lọc cột đặc trưng
        df_transformed = transformer.transform_df(df)
        recent_features = df_transformed[transformer.feature_cols].tail(LOOKBACK_WINDOW)
        
        if len(recent_features) < LOOKBACK_WINDOW:
            print(f"❌ Lỗi: Không đủ dữ liệu {LOOKBACK_WINDOW} ngày để tạo sliding window cho {ticker}.")
            return
            
        raw_df = df.tail(LOOKBACK_WINDOW).copy()
        raw_df.set_index('date', inplace=True)
        
        # Scale inputs
        recent_scaled = scaler_X.transform(recent_features.values)
        X_predict = recent_scaled.reshape(1, LOOKBACK_WINDOW, len(transformer.feature_cols))
        
        # Chạy dummy forward pass để khởi tạo các thuộc tính Functional của Keras 3
        _ = transformer_model(X_predict)
        
        # Dự đoán từ Transformer thô
        trans_pred_raw = transformer_model.predict(X_predict, verbose=0)
        trans_pred_clean = trans_pred_raw[0] if isinstance(trans_pred_raw, (list, tuple)) else trans_pred_raw
        if len(trans_pred_clean.shape) == 1:
            trans_pred_clean = trans_pred_clean.reshape(1, -1)
        trans_return_transformer = scaler_y.inverse_transform(trans_pred_clean)[0]

        # Chạy dự báo từ XGBoost Stacking
        X_pred_today = X_predict[0, -1, :].reshape(1, -1)
        X_pred_xgb = np.concatenate([trans_pred_clean, X_pred_today], axis=1)
        xgb_pred_scaled = xgb_model.predict(X_pred_xgb)
        trans_return_xgb = scaler_y.inverse_transform(xgb_pred_scaled)[0]
        
        # Thiết lập trans_return_future là dự báo XGBoost cho các phần logic tiếp theo
        trans_return_future = trans_return_xgb
        
        # Kết quả cuối cùng
        last_close = float(raw_df['close'].iloc[-1])
        last_date = raw_df.index[-1].strftime('%Y-%m-%d')
        
        next_dates = []
        curr_date = pd.to_datetime(last_date)
        for h in [1, 2, 3]:
            curr_date = curr_date + pd.tseries.offsets.BDay(1)
            next_dates.append(curr_date.strftime('%Y-%m-%d'))
            
        trans_vals = last_close * (1 + trans_return_future)
        
        last_atr = float(raw_df['atr_14'].iloc[-1])
        risk_ratio = (last_atr / last_close) * 100
        
        if risk_ratio < 1.5:
            risk_level = "Thấp (An toàn - Thị trường ổn định) 🟢"
        elif risk_ratio < 3.0:
            risk_level = "Trung bình (Biến động nhẹ - Thận trọng) 🟡"
        else:
            risk_level = "Cao (Nguy hiểm - Biến động cực mạnh) 🔴"
            
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

        tech_rep = tech_agent.analyze(ticker, trans_return_transformer, trans_return_xgb, rsi_14, macd_ratio, bb_position)
        sent_rep = sent_agent.analyze(ticker, news_sentiment_score)
        macro_rep = macro_agent.analyze(ticker, vix_lag1, bond_yield_lag1, usdvnd_change, vnindex_return_lag1)
        risk_rep = risk_agent.analyze(ticker, last_close, last_atr, mfi_14)

        decision = orchestrator.run_debate(ticker, last_close, tech_rep, sent_rep, macro_rep, risk_rep)
        
        print("==========================================================================")
        
        # Ghi nhận kết quả dự đoán vào tệp history log với nhãn mô hình rõ ràng
        logs_dir = os.path.join(ROOT_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "predict_predictions_history.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # In kết quả trực tiếp ra màn hình cho thân thiện
        print(f"📊 KẾT QUẢ DỰ BÁO 3 NGÀY CHO MÃ: {ticker}")
        print(f"  💵 Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ")
        print(f"  ⚠️  Mức độ rủi ro biến động: {risk_level} ({risk_ratio:.2f}%)")
        if "VNM" not in ticker.upper():
            print(f"  💵 Quy đổi USD: ${last_close/USD_TO_VND:,.2f} USD (Tỷ giá: {format_vn(USD_TO_VND)} VNĐ)")
            print("  🌳 Dự báo Hybrid XGBoost (Chuỗi 3 ngày):")
            for h in range(3):
                trend = f"📈 TĂNG ({trans_return_future[h]*100:+.2f}%)" if trans_vals[h] >= last_close else f"📉 GIẢM ({trans_return_future[h]*100:+.2f}%)"
                print(f"     ➔ T+{h+1} ({next_dates[h]}): {format_vn(trans_vals[h])} VNĐ (${trans_vals[h]/USD_TO_VND:.2f} USD) | {trend}")
        else:
            print("  🌳 Dự báo Hybrid XGBoost (Chuỗi 3 ngày):")
            for h in range(3):
                trend = f"📈 TĂNG ({trans_return_future[h]*100:+.2f}%)" if trans_vals[h] >= last_close else f"📉 GIẢM ({trans_return_future[h]*100:+.2f}%)"
                print(f"     ➔ T+{h+1} ({next_dates[h]}): {format_vn(trans_vals[h])} VNĐ | {trend}")

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
        
        # Tải win_rate_history từ file cấu hình backtest (mặc định 0.50)
        win_rate_history = 0.50
        perf_path = os.path.join(ROOT_DIR, 'config', f'performance_metrics_{ticker}.json')
        if os.path.exists(perf_path):
            try:
                with open(perf_path, 'r', encoding='utf-8') as f:
                    perf_data = json.load(f)
                win_rate_history = perf_data.get('overall_win_rate', perf_data.get('win_rate', 0.50))
                print(f"🥇 Nạp win_rate_history thành công cho {ticker}: {win_rate_history*100:.2f}%")
            except Exception as e:
                print(f"⚠️ Không nạp được file performance metrics cho {ticker}: {e}")

        # Tính toán Kelly Criterion động
        kelly_size_pct = 0.0
        kelly_raw = 0.0
        kelly_half = 0.0
        R_val = 2.00
        p_val = win_rate_history
        
        if decision.action == "BUY":
            kelly_results = risk_agent.calculate_position_size(
                confidence_score=decision.confidence_score,
                win_rate_history=win_rate_history,
                close_price=last_close,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit
            )
            kelly_size_pct = kelly_results["kelly_size_pct"]
            kelly_raw = kelly_results["kelly_raw"]
            kelly_half = kelly_results["kelly_half"]
            R_val = kelly_results["R"]
            p_val = kelly_results["p"]

        if decision.action == "BUY":
            print(f"   ➔ Cắt lỗ (SL)  : {format_vn(decision.stop_loss)} VNĐ ({loss_pct:+.2f}%)")
            print(f"   ➔ Chốt lời (TP): {format_vn(decision.take_profit)} VNĐ ({profit_pct:+.2f}%)")
            print(f"   ➔ Tỷ lệ R/R    : 1:{R_val:.2f}")
            print(f"   ➔ Xác suất Kelly: {p_val*100:.2f}% (Orchestrator: {decision.confidence_score*100:.1f}%, Lịch sử: {win_rate_history*100:.2f}%)")
            print(f"   ➔ Phân bổ vốn  : {kelly_size_pct:.2f}% tài khoản (Half-Kelly, Cap 25%)")
        else:
            print(f"   ➔ SL tham chiếu: {format_vn(ref_sl)} VNĐ ({loss_pct:+.2f}%)")
            print(f"   ➔ TP tham chiếu: {format_vn(ref_tp)} VNĐ ({profit_pct:+.2f}%)")
            print(f"   ➔ Tỷ lệ R/R    : 1:2.00")
            print(f"   ➔ Phân bổ vốn  : 0.00% tài khoản (Khuyến nghị HOLD/SELL)")
        print(f"   ➔ Lập luận    : {decision.reasoning}")
        print("==========================================================================")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== BẢN GHI DỰ BÁO 3 NGÀY & MULTI-AGENT REPORT ({timestamp}) ===\n")
            f.write(f"Mã chứng khoán: {ticker}\n")
            if "VNM" in ticker.upper():
                f.write(f"Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ\n")
                f.write(f"Rủi ro biến động: {risk_level} (Tỷ lệ: {risk_ratio:.2f}%)\n")
                f.write("Dự báo Hybrid XGBoost (3 ngày):\n")
                for h in range(3):
                    trend = f"📈 TĂNG ({trans_return_future[h]*100:+.2f}%)" if trans_vals[h] >= last_close else f"📉 GIẢM ({trans_return_future[h]*100:+.2f}%)"
                    f.write(f"  ➔ T+{h+1} ({next_dates[h]}): {format_vn(trans_vals[h])} VNĐ ({trend}) | Khoảng an toàn: [{format_vn(trans_lowers[h])} - {format_vn(trans_uppers[h])}] VNĐ\n")
            else:
                f.write(f"Giá đóng cửa gần nhất ({last_date}): {format_vn(last_close)} VNĐ (${last_close/USD_TO_VND:,.2f} USD)\n")
                f.write(f"Rủi ro biến động: {risk_level} (Tỷ lệ: {risk_ratio:.2f}%)\n")
                f.write(f"Tỷ giá USD/VND quy đổi: 1 USD = {format_vn(USD_TO_VND)} VNĐ\n")
                f.write("Dự báo Hybrid XGBoost (3 ngày):\n")
                for h in range(3):
                    trend = f"📈 TĂNG ({trans_return_future[h]*100:+.2f}%)" if trans_vals[h] >= last_close else f"📉 GIẢM ({trans_return_future[h]*100:+.2f}%)"
                    f.write(f"  ➔ T+{h+1} ({next_dates[h]}): {format_vn(trans_vals[h])} VNĐ (${trans_vals[h]/USD_TO_VND:.2f} USD) ({trend}) | Khoảng an toàn: [{format_vn(trans_lowers[h])} - {format_vn(trans_uppers[h])}] VNĐ\n")
            f.write("\n🤖 BÁO CÁO MULTI-AGENT:\n")
            f.write(f"Khuyến nghị cuối cùng: {decision.action} (Độ tự tin: {decision.confidence_score*100:.1f}%)\n")
            if decision.action == "BUY":
                f.write(f"  * Cắt lỗ (SL)  : {format_vn(decision.stop_loss)} VNĐ ({loss_pct:+.2f}%)\n")
                f.write(f"  * Chốt lời (TP): {format_vn(decision.take_profit)} VNĐ ({profit_pct:+.2f}%)\n")
                f.write(f"  * Phân bổ Kelly: {kelly_size_pct:.2f}% (Half-Kelly, Cap 25%)\n")
            else:
                f.write(f"  * SL tham chiếu: {format_vn(ref_sl)} VNĐ ({loss_pct:+.2f}%)\n")
                f.write(f"  * TP tham chiếu: {format_vn(ref_tp)} VNĐ ({profit_pct:+.2f}%)\n")
                f.write(f"  * Phân bổ Kelly: 0.00%\n")
            f.write(f"Lập luận: {decision.reasoning}\n")
            f.write("-" * 50 + "\n\n")
            
        print("💾 Đã tự động ghi nhận kết quả dự đoán và báo cáo Multi-Agent vào nhật ký lịch sử.")

        # Gửi tin nhắn Telegram thông báo kết quả và khuyến nghị Multi-Agent
        try:
            from src.notifications import send_telegram_message
            
            usd_desc = f" (${last_close/USD_TO_VND:,.2f} USD)" if "VNM" not in ticker.upper() else ""
            
            def trend_str(val):
                pct = (val - last_close) / last_close * 100
                return f"{'📈 TĂNG' if val >= last_close else '📉 GIẢM'} ({pct:+.2f}%)"
            
            # Ngưỡng biến động mạnh động cho từng ticker
            ALERT_THRESHOLD = {
                'VNM.VN': 0.03,   # 3%
                'GOOGL':  0.025,  # 2.5%
                'META':   0.025,  # 2.5%
            }
            threshold = ALERT_THRESHOLD.get(ticker, 0.03)
            
            # Tính toán thay đổi dự báo lớn nhất
            max_forecast_change = max([abs(ret) for ret in trans_return_future]) * 100.0
            
            # In debug log lên console
            print(f"🔍 [Telegram] Kiểm tra biến động mạnh cho {ticker}: ATR = {risk_ratio:.2f}% (ngưỡng {threshold * 100:.1f}%), Dự báo thay đổi lớn nhất = {max_forecast_change:.2f}% (ngưỡng {threshold * 100:.1f}%).")
            
            # Kiểm tra biến động mạnh
            is_strong_volatility = False
            volatility_reasons = []
            if risk_ratio >= (threshold * 100.0):
                is_strong_volatility = True
                volatility_reasons.append(f"Mức rủi ro cao ({risk_ratio:.2f}%)")
            for h in range(3):
                ret = trans_return_future[h]
                if abs(ret) >= threshold:
                    is_strong_volatility = True
                    direction = "TĂNG" if ret > 0 else "GIẢM"
                    volatility_reasons.append(f"Dự báo {direction} mạnh T+{h+1} ({ret*100:+.2f}%)")

            # Xác định các nhân tố tác động (Catalysts)
            catalysts = []
            
            # 1. Kiểm tra tin tức nóng (Breaking News hoặc Tin tiêu cực/tích cực mạnh)
            from src.news_sentiment import fetch_latest_news
            try:
                news_items = fetch_latest_news(ticker)
                breaking_titles = [n['title'] for n in news_items if "BREAKING" in n['title'].upper() or "SẬP" in n['title'].upper() or "OUTAGE" in n['title'].upper()]
                if breaking_titles:
                    title = breaking_titles[0]
                    if "outage" in title.lower() or "sập" in title.lower():
                        catalysts.append("🔴 <b>Tin tức nóng:</b> Sự cố sập hệ thống toàn cầu của Meta gây sụt giảm doanh thu quảng cáo nghiêm trọng.")
                    else:
                        catalysts.append(f"🔴 <b>Tin tức nóng:</b> {title}")
                elif abs(news_sentiment_score) > 0.4:
                    recent_titles = [n['title'] for n in news_items[:2]]
                    for rt in recent_titles:
                        prefix = "🟢" if news_sentiment_score > 0 else "🔴"
                        catalysts.append(f"{prefix} <b>Tin tức nổi bật:</b> {rt}")
            except Exception:
                pass

            # 2. Kiểm tra dòng tiền / khối lượng (MFI và RSI)
            if rsi_14 > 80 or mfi_14 > 80:
                catalysts.append(f"📈 <b>Dòng tiền:</b> Lực mua đột biến quá mức (Quá mua) - RSI: <b>{rsi_14:.1f}</b>, MFI: <b>{mfi_14:.1f}</b>. Có dòng tiền lớn đang gom hàng đẩy giá.")
            elif rsi_14 < 25 or mfi_14 < 25:
                catalysts.append(f"📉 <b>Dòng tiền:</b> Lực bán tháo hoảng loạn cực mạnh (Quá bán) - RSI: <b>{rsi_14:.1f}</b>, MFI: <b>{mfi_14:.1f}</b>. Có áp lực tháo chạy khỏi cổ phiếu.")

            catalysts_desc = ""
            if catalysts:
                catalysts_desc = "🔥 <b>NHÂN TỐ BIẾN ĐỘNG / CATALYSTS:</b>\n" + "\n".join([f"• {c}" for c in catalysts]) + "\n\n"

            warning_header = ""
            if is_strong_volatility:
                warning_header = (
                    f"⚠️⚠️⚠️ <b>CẢNH BÁO BIẾN ĐỘNG MẠNH</b> ⚠️⚠️⚠️\n"
                    f"📌 Lý do: {', '.join(volatility_reasons)}\n"
                    f"-----------------------------------------\n"
                )

            forecast_rows = ""
            for h in range(3):
                usd_val_text = f" (${trans_vals[h]/USD_TO_VND:.2f} USD)" if "VNM" not in ticker.upper() else ""
                forecast_rows += f"• T+{h+1} ({next_dates[h]}): <b>{format_vn(trans_vals[h])} VNĐ</b>{usd_val_text} | {trend_str(trans_vals[h])}\n"
                
            sl_tp_desc = ""
            if decision.action == "BUY":
                sl_tp_desc = (
                    f"Cắt lỗ (SL)  : <b>{format_vn(decision.stop_loss)} VNĐ</b> ({loss_pct:+.2f}%)\n"
                    f"Chốt lời (TP): <b>{format_vn(decision.take_profit)} VNĐ</b> ({profit_pct:+.2f}%)\n"
                    f"Phân bổ Kelly: <b>{kelly_size_pct:.2f}%</b> (Half-Kelly, Cap 25%)"
                )
            else:
                sl_tp_desc = (
                    f"SL tham chiếu: <b>{format_vn(ref_sl)} VNĐ</b> ({loss_pct:+.2f}%)\n"
                    f"TP tham chiếu: <b>{format_vn(ref_tp)} VNĐ</b> ({profit_pct:+.2f}%)\n"
                    f"Phân bổ Kelly: <b>0.00%</b>"
                )
                
            action_emoji = "🟢" if decision.action == "BUY" else "🔴" if decision.action == "SELL" else "🟡"
            
            telegram_msg = (
                f"{warning_header}"
                f"📊 <b>KẾT QUẢ DỰ BÁO GIÁ & MULTI-AGENT - {ticker}</b>\n"
                f"-----------------------------------------\n"
                f"💵 Giá đóng cửa gần nhất ({last_date}): <b>{format_vn(last_close)} VNĐ</b>{usd_desc}\n"
                f"⚠️ Mức độ rủi ro: {risk_level} ({risk_ratio:.2f}%)\n\n"
                f"{catalysts_desc}"
                f"🌳 <b>Dự báo Hybrid XGBoost (3 ngày):</b>\n"
                f"{forecast_rows}\n"
                f"🤖 <b>Quyết định đầu tư Multi-Agent:</b>\n"
                f"Khuyến nghị: {action_emoji} <b>{decision.action}</b> (Độ tự tin: {decision.confidence_score*100:.1f}%)\n"
                f"{sl_tp_desc}\n\n"
                f"💬 <b>Lập luận:</b>\n"
                f"<i>{decision.reasoning}</i>"
            )
            if is_strong_volatility:
                send_telegram_message(telegram_msg)
            else:
                print(f"ℹ️ [Telegram] Biến động bình thường ({risk_ratio:.2f}%), không gửi thông báo Telegram.")
        except Exception as te:
            print(f"⚠️ [Telegram] Không gửi được thông báo: {te}")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi tính toán dự đoán: {e}")
        import traceback
        traceback.print_exc()

def main():
    target = "VNM.VN"
    if len(sys.argv) > 1:
        target = sys.argv[1].upper()
    
    TICKERS = ["VNM.VN", "GOOGL", "META"]
    if target == "ALL":
        for t in TICKERS:
            try:
                run_prediction_for_ticker(t)
            except Exception as e:
                print(f"❌ Lỗi khi dự đoán cho mã {t}: {e}")
    else:
        run_prediction_for_ticker(target)

if __name__ == "__main__":
    main()
