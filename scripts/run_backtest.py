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
# run_backtest.py — ĐÃ SỬA CÁC LỖI:
# 1. volume_zscore KeyError: tính động từ df_test['volume'] thay vì đọc column không tồn tại
# 2. Win rate tính sai: đổi sang zip(buy_trades, sell_trades) theo thứ tự thời gian
# 3. Xử lý vị thế mở cuối cùng (open position khi hết data)
# 4. get_return_output() helper để handle list output của Transformer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data, format_vn
from src.features import DataTransformer
from src.ai_models import (
    PositionalEmbedding, TimeDecayAttention,
    MultiTaskModel, UncertaintyWeightsLayer,
)


def get_return_output(pred):
    """Handle Transformer output: list [return, spread] hoặc tensor đơn."""
    return pred[0] if isinstance(pred, (list, tuple)) else pred


def compute_metrics(equity, bh_equity, dates):
    equity    = np.array(equity)
    bh_equity = np.array(bh_equity)
    total_strat_ret = (equity[-1] - equity[0]) / equity[0] * 100
    total_bh_ret    = (bh_equity[-1] - bh_equity[0]) / bh_equity[0] * 100

    strat_daily_ret = np.diff(equity) / equity[:-1]
    if len(strat_daily_ret) > 1 and np.std(strat_daily_ret) > 1e-8:
        sharpe = np.sqrt(252) * (np.mean(strat_daily_ret) / np.std(strat_daily_ret))
    else:
        sharpe = 0.0

    peaks = np.maximum.accumulate(equity)
    mdd   = np.min((equity - peaks) / peaks) * 100

    bh_peaks = np.maximum.accumulate(bh_equity)
    bh_mdd   = np.min((bh_equity - bh_peaks) / bh_peaks) * 100

    return {
        "strat_return": total_strat_ret,
        "bh_return":    total_bh_ret,
        "sharpe":       sharpe,
        "mdd":          mdd,
        "bh_mdd":       bh_mdd,
    }


def run_simulation(
    df_test, X_test, y_test_raw, dt, xgb_model, transformer_model,
    ticker, commission_pct, slippage_pct,
    threshold_buy=0.0010, threshold_sell=-0.0010,
):
    feature_extractor = Model(
        inputs=transformer_model.input,
        outputs=transformer_model.get_layer("latent_embedding").output,
    )

    print("   [PREDICT] Đang chạy dự báo trên tập test...")
    X_test_latent  = feature_extractor.predict(X_test, verbose=0)
    X_test_today   = X_test[:, -1, :]
    X_test_hybrid  = np.concatenate([X_test_latent, X_test_today], axis=1)

    xgb_pred_scaled = xgb_model.predict(X_test_hybrid).reshape(-1, 1)
    xgb_returns     = dt.target_scaler.inverse_transform(xgb_pred_scaled).ravel()

    # SỬA: dùng get_return_output() để handle list output
    trans_pred_raw   = transformer_model.predict(X_test, verbose=0)
    trans_pred_clean = get_return_output(trans_pred_raw)
    trans_returns    = dt.target_scaler.inverse_transform(trans_pred_clean).ravel()

    # SỬA: sanity check — return > 20%/ngày là scaler sai
    assert np.abs(xgb_returns).max() < 0.20, \
        f"[SANITY] XGBoost return bất thường: {np.abs(xgb_returns).max():.4f}"
    assert np.abs(trans_returns).max() < 0.20, \
        f"[SANITY] Transformer return bất thường: {np.abs(trans_returns).max():.4f}"

    close_today   = df_test['close'].values[:len(X_test)]
    open_tomorrow = df_test['open'].shift(-1).ffill().bfill().values[:len(X_test)]
    dates         = df_test['date'].values[:len(X_test)]

    # === SỬA: Tính volume_zscore động từ df_test — tránh KeyError ===
    # Trước: df_test['volume_zscore'] → KeyError vì column này không có trong df gốc
    # Sau: tính tại chỗ từ volume
    vol_series  = df_test['volume'].values[:len(X_test)]
    vol_roll_n  = 20
    vol_mean    = pd.Series(vol_series).rolling(vol_roll_n, min_periods=1).mean().values
    vol_std     = pd.Series(vol_series).rolling(vol_roll_n, min_periods=1).std().fillna(1.0).values
    volume_zscore_arr = (vol_series - vol_mean) / (vol_std + 1e-9)

    initial_capital = 100_000_000.0
    cash    = initial_capital
    shares  = 0.0
    position = 0

    equity    = [initial_capital]
    bh_shares = initial_capital / close_today[0]
    bh_equity = [initial_capital]
    trades    = []

    for i in range(len(X_test) - 1):
        t_close      = close_today[i]
        t_open_next  = open_tomorrow[i]
        r_xgb        = xgb_returns[i]
        r_trans      = trans_returns[i]

        sig_xgb   = "Up" if r_xgb > threshold_buy else ("Down" if r_xgb < threshold_sell else "Neutral")
        sig_trans = "Up" if r_trans > threshold_buy else ("Down" if r_trans < threshold_sell else "Neutral")
        is_buy_signal = (sig_xgb == "Up" and sig_trans == "Up")

        # SỬA: vol_zscore từ mảng đã tính sẵn
        current_slippage = slippage_pct * 2.0 if volume_zscore_arr[i] < -1.0 else slippage_pct

        buy_price  = t_close     * (1 + current_slippage)
        sell_price = t_open_next * (1 - current_slippage)

        date_str = lambda d: d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)

        # Đóng vị thế cũ (sáng hôm sau)
        if position == 1:
            cash = shares * sell_price * (1 - commission_pct)
            shares = 0.0
            position = 0
            trades.append({
                "date": date_str(dates[i + 1]),
                "type": "SELL",
                "price": sell_price,
                "cash": cash,
            })

        # Mở vị thế mới (chiều hôm nay)
        if is_buy_signal:
            shares = (cash * (1 - commission_pct)) / buy_price
            cash = 0.0
            position = 1
            trades.append({
                "date":   date_str(dates[i]),
                "type":   "BUY",
                "price":  buy_price,
                "shares": shares,
            })

        current_equity = (
            cash if position == 0
            else shares * t_open_next * (1 - current_slippage) * (1 - commission_pct)
        )
        equity.append(current_equity)
        bh_equity.append(bh_shares * t_open_next)

    # === SỬA: xử lý vị thế mở cuối giai đoạn ===
    if position == 1:
        final_price = close_today[-1] * (1 - slippage_pct)
        cash = shares * final_price * (1 - commission_pct)
        trades.append({
            "date": date_str(dates[-1]), "type": "SELL",
            "price": final_price, "cash": cash,
        })

    return dates, equity, bh_equity, trades


def compute_win_rate(trades):
    """
    === SỬA: Tính win rate đúng ===
    Trước: loop step=2 theo index chẵn/lẻ → sai nếu trades lẻ hoặc không xen kẽ đều
    Sau: tách riêng buy/sell list, zip theo thứ tự thời gian, tính P&L từng cặp
    """
    buy_trades  = [t for t in trades if t["type"] == "BUY"]
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    total = min(len(buy_trades), len(sell_trades))

    if total == 0:
        return 0.0, 0

    win_count = 0
    for buy, sell in zip(buy_trades[:total], sell_trades[:total]):
        # P&L = sell_cash - buy_cost (phí đã tính trong cash)
        buy_cost = buy['price'] * buy['shares']
        if sell['cash'] > buy_cost:
            win_count += 1

    return (win_count / total * 100), total


def run_walk_forward_evaluation(dates, equity, bh_equity):
    total_len   = len(dates)
    window_size = total_len // 3
    results = []
    for w in range(3):
        start_idx = w * window_size
        end_idx   = total_len if w == 2 else (w + 1) * window_size
        metrics   = compute_metrics(equity[start_idx:end_idx], bh_equity[start_idx:end_idx], dates[start_idx:end_idx])
        d_start   = pd.to_datetime(dates[start_idx]).strftime('%Y-%m-%d')
        d_end     = pd.to_datetime(dates[end_idx - 1]).strftime('%Y-%m-%d')
        results.append({"window": f"Window {w+1} ({d_start} → {d_end})", **metrics})
    return results


def main():
    TICKERS = ["VNM.VN", "GOOGL", "META"]
    threshold_buy = 0.0010

    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
    if len(sys.argv) > 2:
        try:
            threshold_buy = float(sys.argv[2])
        except ValueError:
            pass

    threshold_sell = -threshold_buy
    models_dir  = os.path.join(ROOT_DIR, 'models')
    figures_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'='*70}")
        print(f"📊 BACKTEST: {ticker}")
        print(f"{'='*70}")

        if "VNM" in ticker.upper():
            commission_pct = 0.0020
            slippage_pct   = 0.0010
        else:
            commission_pct = 0.0010
            slippage_pct   = 0.0005

        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date="2026-05-20")

        dt = DataTransformer(time_steps=45)
        X_scaled, y_scaled, y_spread_scaled = dt.fit_transform_train_only(df, train_ratio=0.8)
        X_3D, y_3D, y_spread_3D = dt.create_sliding_windows(X_scaled, y_scaled, y_spread_scaled)

        X_train, y_train, X_test, y_test, y_test_raw, y_train_spread, y_test_spread = \
            dt.split_train_test_chronological(df, X_3D, y_3D, y_spread_3D, train_ratio=0.8)

        df_align      = df.iloc[dt.time_steps:].reset_index(drop=True)
        split_idx     = int(len(X_3D) * 0.8)
        test_start_idx = split_idx + 45
        df_test = df_align.iloc[test_start_idx:].reset_index(drop=True)

        xgb_path   = os.path.join(models_dir, f'xgboost_model_{ticker}.pkl')
        trans_path = os.path.join(models_dir, f'transformer_model_{ticker}.keras')
        feat_path  = os.path.join(models_dir, f'feature_scaler_{ticker}.pkl')
        targ_path  = os.path.join(models_dir, f'target_scaler_{ticker}.pkl')

        if not (os.path.exists(xgb_path) and os.path.exists(trans_path)):
            print(f"❌ Không tìm thấy model cho {ticker}. Chạy run_training.py trước.")
            continue

        xgb_model = joblib.load(xgb_path)
        transformer_model = tf.keras.models.load_model(
            trans_path,
            custom_objects={
                'PositionalEmbedding': PositionalEmbedding,
                'TimeDecayAttention': TimeDecayAttention,
                'MultiTaskModel': MultiTaskModel,
                'UncertaintyWeightsLayer': UncertaintyWeightsLayer,
            },
            safe_mode=False
        )
        dt.feature_scaler = joblib.load(feat_path)
        dt.target_scaler  = joblib.load(targ_path)

        dates, equity, bh_equity, trades = run_simulation(
            df_test, X_test, y_test_raw, dt,
            xgb_model, transformer_model, ticker,
            commission_pct, slippage_pct, threshold_buy, threshold_sell,
        )

        metrics = compute_metrics(equity, bh_equity, dates)

        # SỬA: win rate tính đúng
        win_rate, total_trades = compute_win_rate(trades)

        print(f"\n🏆 KẾT QUẢ BACKTEST (Out-of-Sample):")
        print(f"   📈 Chiến lược: {metrics['strat_return']:+.2f}%")
        print(f"   📦 Buy&Hold : {metrics['bh_return']:+.2f}%")
        print(f"   📊 Sharpe   : {metrics['sharpe']:.2f}")
        print(f"   📉 Max DD   : {metrics['mdd']:.2f}% (B&H: {metrics['bh_mdd']:.2f}%)")
        print(f"   🔔 Số lệnh  : {total_trades}")
        print(f"   🥇 Win Rate : {win_rate:.2f}%")

        print(f"\n🎯 WALK-FORWARD (3 Windows):")
        wf = run_walk_forward_evaluation(dates, equity, bh_equity)
        print("-" * 95)
        print(f"{'Window':<40} | {'Strategy':<12} | {'B&H':<12} | {'Sharpe':<8} | {'MDD':<8}")
        print("-" * 95)
        for w in wf:
            print(f"{w['window']:<40} | {w['strat_return']:+11.2f}% | {w['bh_return']:+11.2f}% | {w['sharpe']:<8.2f} | {w['mdd']:<8.2f}%")
        print("-" * 95)

        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity,    label="Hybrid Strategy", color='darkgreen', linewidth=2)
        plt.plot(dates, bh_equity, label="Buy & Hold",      color='grey', linestyle='--', alpha=0.8)
        plt.title(f"Equity Curve — {ticker}", fontsize=13, fontweight='bold')
        plt.xlabel("Ngày"); plt.ylabel("Tài sản (VNĐ)")
        plt.legend(); plt.grid(True, alpha=0.25)
        plot_path = os.path.join(figures_dir, f'backtest_equity_curve_{ticker}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 Equity curve: {plot_path}")


if __name__ == "__main__":
    main()