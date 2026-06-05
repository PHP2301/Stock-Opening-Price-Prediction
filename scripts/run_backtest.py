import os
import sys
import datetime
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

# Cố định random seed để kết quả tái lập tốt nhất
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thiết lập đường dẫn thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data, format_vn
from src.features import DataTransformer
from src.ai_models import PositionalEmbedding, TimeDecayAttention, MultiTaskModel, UncertaintyWeightsLayer

def compute_metrics(equity, bh_equity, dates):
    """
    Tính toán các chỉ số đo lường hiệu suất giao dịch nâng cao:
    - Sharpe Ratio
    - Maximum Drawdown (MDD)
    - Total Return (%)
    """
    equity = np.array(equity)
    bh_equity = np.array(bh_equity)
    
    # 1. Tỷ suất lợi nhuận tích lũy (Total Return)
    total_strat_ret = (equity[-1] - equity[0]) / equity[0] * 100
    total_bh_ret = (bh_equity[-1] - bh_equity[0]) / bh_equity[0] * 100
    
    # 2. Sharpe Ratio (Annualized)
    # Lợi nhuận hàng ngày
    strat_daily_ret = np.diff(equity) / equity[:-1]
    if len(strat_daily_ret) > 1 and np.std(strat_daily_ret) > 1e-8:
        # Giả định Risk-Free Rate = 0
        sharpe = np.sqrt(252) * (np.mean(strat_daily_ret) / np.std(strat_daily_ret))
    else:
        sharpe = 0.0
        
    # 3. Maximum Drawdown (MDD)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    mdd = np.min(drawdowns) * 100  # phần trăm âm nhất
    
    # MDD của Buy & Hold
    bh_peaks = np.maximum.accumulate(bh_equity)
    bh_drawdowns = (bh_equity - bh_peaks) / bh_peaks
    bh_mdd = np.min(bh_drawdowns) * 100
    
    return {
        "strat_return": total_strat_ret,
        "bh_return": total_bh_ret,
        "sharpe": sharpe,
        "mdd": mdd,
        "bh_mdd": bh_mdd
    }

def run_simulation(df_test, X_test, y_test_raw, transformer, xgb_model, transformer_model, ticker, commission_pct, slippage_pct, threshold_buy=0.0010, threshold_sell=-0.0010):
    """
    Giả lập giao dịch qua đêm (Overnight Trading):
    - Đưa ra dự đoán tại Close ngày hôm nay.
    - Nếu cả hai mô hình (XGBoost Lai & Transformer) đồng thuận BÁO TĂNG (> threshold_buy): MUA tại Close (chịu phí + trượt giá).
    - BÁN toàn bộ tại Open ngày hôm sau (chịu phí + trượt giá).
    """
    # Xây dựng bộ trích xuất đặc trưng ẩn chính từ Transformer
    feature_extractor = Model(
        inputs=transformer_model.input,
        outputs=transformer_model.get_layer("latent_embedding").output
    )
    
    print("   [PREDICT] Đang chạy dự báo trên tập dữ liệu kiểm thử...")
    X_test_latent = feature_extractor.predict(X_test, verbose=0)
    X_test_today = X_test[:, -1, :]
    X_test_hybrid = np.concatenate([X_test_latent, X_test_today], axis=1)
    
    # Dự báo tỷ suất sinh lợi từ 2 mô hình
    xgb_pred_scaled = xgb_model.predict(X_test_hybrid).reshape(-1, 1)
    xgb_returns = transformer.target_scaler.inverse_transform(xgb_pred_scaled).ravel()
    
    trans_pred_output = transformer_model.predict(X_test, verbose=0)
    trans_scaled = trans_pred_output[0] if isinstance(trans_pred_output, list) else trans_pred_output
    trans_returns = transformer.target_scaler.inverse_transform(trans_scaled).ravel()
    
    # Khớp dữ liệu
    close_today = df_test['close'].values[:len(X_test)]
    open_tomorrow = df_test['open'].shift(-1).ffill().bfill().values[:len(X_test)]
    dates = df_test['date'].values[:len(X_test)]
    
    # Khởi tạo giả lập vốn
    initial_capital = 100000000.0  # 100 triệu VNĐ làm gốc
    cash = initial_capital
    shares = 0.0
    position = 0  # 0: Cash, 1: Long
    
    equity = [initial_capital]
    bh_shares = initial_capital / close_today[0]
    bh_equity = [initial_capital]
    
    trades = []
    
    for i in range(len(X_test) - 1):
        t_close = close_today[i]
        t_open_next = open_tomorrow[i]
        
        # Dự báo của 2 mô hình
        r_xgb = xgb_returns[i]
        r_trans = trans_returns[i]
        
        # Phân tích Tín hiệu Hướng đi (Direction Signal) & Điểm Tự Tin (Confidence Score)
        sig_xgb = "Up" if r_xgb > threshold_buy else ("Down" if r_xgb < threshold_sell else "Neutral")
        sig_trans = "Up" if r_trans > threshold_buy else ("Down" if r_trans < threshold_sell else "Neutral")
        
        # Quy tắc: Đồng thuận tăng -> MUA qua đêm. Mọi trường hợp khác -> BÁN ra tiền mặt.
        is_buy_signal = (sig_xgb == "Up" and sig_trans == "Up")
        confidence = "High" if (sig_xgb == sig_trans) else "Low"
        
        # Tính trượt giá động (Dynamic Slippage) dựa trên độ thanh khoản của phiên ngày t
        current_slippage = slippage_pct
        vol_z = df_test['volume_zscore'].values[i]
        if vol_z < -1.0:
            current_slippage = slippage_pct * 2.0  # Phạt trượt giá gấp đôi khi thanh khoản quá thấp
            
        buy_price = t_close * (1 + current_slippage)
        sell_price = t_open_next * (1 - current_slippage)
        
        # Sáng hôm sau: Đóng vị thế qua đêm (nếu có vị thế Long)
        if position == 1:
            cash = shares * sell_price * (1 - commission_pct)
            shares = 0.0
            position = 0
            trades.append({
                "date": dates[i+1].strftime('%Y-%m-%d') if hasattr(dates[i+1], 'strftime') else str(dates[i+1]),
                "type": "SELL",
                "price": sell_price,
                "cash": cash
            })
            
        # Chiều hôm nay: Mở vị thế mới nếu có tín hiệu MUA mạnh đồng thuận
        if is_buy_signal:
            shares = (cash * (1 - commission_pct)) / buy_price
            cash = 0.0
            position = 1
            trades.append({
                "date": dates[i].strftime('%Y-%m-%d') if hasattr(dates[i], 'strftime') else str(dates[i]),
                "type": "BUY",
                "price": buy_price,
                "shares": shares
            })
            
        # Ghi nhận giá trị tài sản ròng cuối ngày t+1 (Sử dụng current_slippage khớp lệnh để cập nhật định giá định kỳ)
        current_equity = cash if position == 0 else shares * t_open_next * (1 - current_slippage) * (1 - commission_pct)
        equity.append(current_equity)
        
        # Buy & Hold
        bh_current = bh_shares * t_open_next
        bh_equity.append(bh_current)
        
    return dates, equity, bh_equity, trades

def run_walk_forward_evaluation(dates, equity, bh_equity):
    """
    Thực hiện đánh giá Walk-forward Validation bằng cách chia tập Test thành 3 Rolling Windows độc lập
    và báo cáo hiệu suất chi tiết trên từng giai đoạn.
    """
    total_len = len(dates)
    window_size = total_len // 3
    
    results = []
    for w in range(3):
        start_idx = w * window_size
        end_idx = total_len if w == 2 else (w + 1) * window_size
        
        w_dates = dates[start_idx:end_idx]
        w_eq = equity[start_idx:end_idx]
        w_bh = bh_equity[start_idx:end_idx]
        
        metrics = compute_metrics(w_eq, w_bh, w_dates)
        results.append({
            "window": f"Window {w+1} ({pd.to_datetime(w_dates[0]).strftime('%Y-%m-%d')} -> {pd.to_datetime(w_dates[-1]).strftime('%Y-%m-%d')})",
            **metrics
        })
    return results

def main():
    # Watchlist tickers mặc định
    TICKERS = ["VNM.VN", "GOOGL", "META"]
    
    threshold_buy = 0.0010  # Mặc định +0.10% thay vì +0.25% để tránh quá thận trọng
    
    # Cho phép chọn ticker cụ thể qua command line
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
            print(f"🎯 Chỉ chạy Backtest mô phỏng cho mã: {TICKERS[0]}")
            
    # Cho phép điều chỉnh ngưỡng kích hoạt lệnh từ tham số dòng lệnh thứ 2 (ví dụ: 0.0025)
    if len(sys.argv) > 2:
        try:
            threshold_buy = float(sys.argv[2])
            print(f"⚙️ Sử dụng ngưỡng kích hoạt lệnh tùy chỉnh: {threshold_buy*100:.3f}%")
        except ValueError:
            pass
            
    threshold_sell = -threshold_buy
            
    models_dir = os.path.join(ROOT_DIR, 'models')
    figures_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    for ticker in TICKERS:
        print(f"\n==========================================================================")
        print(f"📊 KHỞI ĐỘNG GIẢ LẬP BACKTEST THỰC TẾ CHO MÃ: {ticker}")
        print(f"==========================================================================")
        
        # Thiết lập phí và độ trượt giá theo thị trường tương ứng
        if "VNM" in ticker.upper():
            # Thị trường Việt Nam: Phí giao dịch + Thuế TNCN khi bán cao hơn
            commission_pct = 0.0020  # 0.20% (bao gồm thuế 0.1% khi bán)
            slippage_pct = 0.0010    # 0.10% (trượt giá khớp lệnh do spread rộng)
        else:
            # Thị trường Mỹ: Phí giao dịch cực thấp, chênh lệch spread nhỏ
            commission_pct = 0.0010  # 0.10%
            slippage_pct = 0.0005    # 0.05%
            
        print(f"⚙️ Tham số Market Friction: Phí giao dịch = {commission_pct*100:.2f}%, Trượt giá = {slippage_pct*100:.3f}%")
        
        # Tải dữ liệu huấn luyện và test
        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date="2026-05-20")
        
        # Tiền xử lý
        transformer = DataTransformer(time_steps=45)
        X_scaled, y_scaled, y_spread_scaled = transformer.fit_transform_train_only(df, train_ratio=0.8)
        X_3D, y_3D, y_spread_3D = transformer.create_sliding_windows(X_scaled, y_scaled, y_spread_scaled)
        
        # Tách 80/20 train/test kết hợp Purge Gap 45 phiên
        X_train, y_train, X_test, y_test, y_test_raw, y_train_spread, y_test_spread = transformer.split_train_test_chronological(
            df, X_3D, y_3D, y_spread_3D, train_ratio=0.8
        )
        
        # Lấy DataFrame tương ứng với tập test đã áp dụng Purge Gap
        df_align = df.iloc[transformer.time_steps:].reset_index(drop=True)
        split_idx = int(len(X_3D) * 0.8)
        test_start_idx = split_idx + 45
        df_test = df_align.iloc[test_start_idx:].reset_index(drop=True)
        
        # Nạp mô hình đã lưu
        xgb_path = os.path.join(models_dir, f'xgboost_model_{ticker}.pkl')
        trans_path = os.path.join(models_dir, f'transformer_model_{ticker}.keras')
        feat_scaler_path = os.path.join(models_dir, f'feature_scaler_{ticker}.pkl')
        targ_scaler_path = os.path.join(models_dir, f'target_scaler_{ticker}.pkl')
        
        if not (os.path.exists(xgb_path) and os.path.exists(trans_path)):
            print(f"❌ [LỖI] Không tìm thấy file mô hình của {ticker} trong models/!")
            print(f"   Vui lòng chạy 'python scripts/run_training.py' trước để huấn luyện và lưu mô hình.")
            continue
            
        # Nạp mô hình & scalers
        xgb_model = joblib.load(xgb_path)
        transformer_model = tf.keras.models.load_model(
            trans_path, 
            custom_objects={
                'PositionalEmbedding': PositionalEmbedding,
                'TimeDecayAttention': TimeDecayAttention,
                'MultiTaskModel': MultiTaskModel,
                'UncertaintyWeightsLayer': UncertaintyWeightsLayer
            }
        )
        transformer.feature_scaler = joblib.load(feat_scaler_path)
        transformer.target_scaler = joblib.load(targ_scaler_path)
        
        # Chạy giả lập trading
        dates, equity, bh_equity, trades = run_simulation(
            df_test, X_test, y_test_raw, transformer, xgb_model, transformer_model,
            ticker, commission_pct, slippage_pct, threshold_buy, threshold_sell
        )
        
        # Tính toán hiệu suất tổng thể
        metrics = compute_metrics(equity, bh_equity, dates)
        
        # Đếm thống kê giao dịch
        buys = [t for t in trades if t["type"] == "BUY"]
        sells = [t for t in trades if t["type"] == "SELL"]
        win_count = 0
        
        # Tính Win Rate
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                t_buy = trades[i]
                t_sell = trades[i+1]
                if t_sell["cash"] > (t_buy["price"] * t_buy["shares"]):
                    win_count += 1
        total_trades = len(buys)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
        
        print("\n🏆 KẾT QUẢ BACKTEST TOÀN GIAI ĐOẠN (OUT-OF-SAMPLE):")
        print(f"   📈 Chiến lược Hybrid (Overnight)  : {metrics['strat_return']:+.2f}%")
        print(f"   📦 Mua & Nắm giữ (Buy & Hold)     : {metrics['bh_return']:+.2f}%")
        print(f"   📊 Tỷ số Sharpe (Chiến lược)      : {metrics['sharpe']:.2f}")
        print(f"   📉 Max Drawdown (Chiến lược)      : {metrics['mdd']:.2f}% (B&H: {metrics['bh_mdd']:.2f}%)")
        print(f"   🔔 Tổng số lệnh giao dịch thực hiện: {total_trades} lệnh mua-bán")
        print(f"   🥇 Tỷ lệ lệnh có lãi (Win Rate)    : {win_rate:.2f}%")
        
        # Đánh giá Walk-Forward OOS
        print("\n🎯 ĐÁNH GIÁ CHI TIẾT WALK-FORWARD VALIDATION (3 WINDOWS CUỐN CHIẾU):")
        wf_results = run_walk_forward_evaluation(dates, equity, bh_equity)
        print("-" * 100)
        print(f"{'Giai đoạn (Window)':<38} | {'Chiến lược (%)':<15} | {'Buy & Hold (%)':<15} | {'Sharpe':<8} | {'MDD (%)':<8}")
        print("-" * 100)
        for w in wf_results:
            print(f"{w['window']:<38} | {w['strat_return']:+14.2f}% | {w['bh_return']:+14.2f}% | {w['sharpe']:<8.2f} | {w['mdd']:<8.2f}%")
        print("-" * 100)
        
        # Vẽ biểu đồ Equity Curve so sánh
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label="Chiến lược Lai (Hybrid Strategy)", color='darkgreen', linewidth=2)
        plt.plot(dates, bh_equity, label="Mua & Nắm giữ (Buy & Hold)", color='grey', linestyle='--', alpha=0.8)
        plt.title(f"BIỂU ĐỒ EQUITY CURVE GIẢ LẬP GIAO DỊCH QUA ĐÊM CHO {ticker}\n(Có tính Phí & Trượt giá thực tế)", fontsize=13, fontweight='bold')
        plt.xlabel("Ngày giao dịch", fontsize=11)
        plt.ylabel("Giá trị Tài sản (VNĐ)", fontsize=11)
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format_vn(x)))
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.25)
        
        plot_path = os.path.join(figures_dir, f'backtest_equity_curve_{ticker}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 Biểu đồ đường cong Equity Curve đã lưu tại: '{plot_path}'")

if __name__ == "__main__":
    main()
