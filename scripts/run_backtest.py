import os, sys, datetime, random, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

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
    total_strat_ret = (equity[-1]    - equity[0])    / equity[0]    * 100
    total_bh_ret    = (bh_equity[-1] - bh_equity[0]) / bh_equity[0] * 100

    strat_daily_ret = np.diff(equity) / equity[:-1]
    sharpe = (
        np.sqrt(252) * np.mean(strat_daily_ret) / np.std(strat_daily_ret)
        if len(strat_daily_ret) > 1 and np.std(strat_daily_ret) > 1e-8
        else 0.0
    )
    peaks    = np.maximum.accumulate(equity)
    mdd      = np.min((equity - peaks) / peaks) * 100
    bh_peaks = np.maximum.accumulate(bh_equity)
    bh_mdd   = np.min((bh_equity - bh_peaks) / bh_peaks) * 100

    return dict(
        strat_return=total_strat_ret,
        bh_return=total_bh_ret,
        sharpe=sharpe, mdd=mdd, bh_mdd=bh_mdd,
    )


def run_simulation(
    df_test, df_test_extended,
    X_test, y_test_raw, dt,
    xgb_model, transformer_model,
    ticker, commission_pct, slippage_pct,
    threshold_buy=0.0010, threshold_sell=-0.0010,
):

    feature_extractor = Model(
        inputs=transformer_model.inputs,
        outputs=transformer_model.get_layer("latent_embedding").output,
    )

    print("   [PREDICT] Đang chạy dự báo trên tập test...")
    X_test_latent = feature_extractor.predict(X_test, verbose=0)
    X_test_today  = X_test[:, -1, :]
    X_test_hybrid = np.concatenate([X_test_latent, X_test_today], axis=1)

    xgb_pred_scaled = xgb_model.predict(X_test_hybrid).reshape(-1, 1)
    xgb_returns     = dt.target_scaler.inverse_transform(xgb_pred_scaled).ravel()

    trans_pred_raw   = transformer_model.predict(X_test, verbose=0)
    trans_pred_clean = get_return_output(trans_pred_raw)
    trans_returns    = dt.target_scaler.inverse_transform(trans_pred_clean).ravel()

    # Sanity check
    assert np.abs(xgb_returns).max() < 0.20, \
        f"[SANITY] XGBoost return bất thường: {np.abs(xgb_returns).max():.4f}"
    assert np.abs(trans_returns).max() < 0.20, \
        f"[SANITY] Transformer return bất thường: {np.abs(trans_returns).max():.4f}"

    # Debug correlation — phát hiện signal inversion
    actual_returns = np.diff(df_test['close'].values[:len(X_test)]) \
                     / (df_test['close'].values[:len(X_test) - 1] + 1e-9)
    corr_xgb   = np.corrcoef(xgb_returns[:-1],  actual_returns)[0, 1]
    corr_trans = np.corrcoef(trans_returns[:-1], actual_returns)[0, 1]
    print(f"   [DEBUG] Corr XGB-Actual={corr_xgb:+.4f} | "
          f"Corr Trans-Actual={corr_trans:+.4f}")
    print(f"   [DEBUG] XGB mean={xgb_returns.mean():.5f} | "
          f"Trans mean={trans_returns.mean():.5f} | "
          f"Actual mean={actual_returns.mean():.5f}")

    close_today = df_test['close'].values[:len(X_test)]
    dates       = df_test['date'].values[:len(X_test)]

    # FIX: open_tomorrow từ buffer row — không dùng shift(-1)+ffill
    open_tomorrow = df_test_extended['open'].values[1:len(X_test) + 1]

    # Volume zscore động
    vol_series = df_test['volume'].values[:len(X_test)]
    vol_mean   = pd.Series(vol_series).rolling(20, min_periods=1).mean().values
    vol_std    = pd.Series(vol_series).rolling(20, min_periods=1).std().fillna(1.0).values
    vol_zscore = (vol_series - vol_mean) / (vol_std + 1e-9)

    initial_capital = 100_000_000.0
    cash     = initial_capital
    shares   = 0.0
    position = 0
    equity    = [initial_capital]
    bh_shares = initial_capital / df_test['open'].values[0]
    bh_equity = [initial_capital]
    trades    = []

    date_str = lambda d: d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)

    for i in range(len(X_test) - 1):
        r_xgb   = xgb_returns[i]
        r_trans = trans_returns[i]

        sig_xgb   = ("Up"   if r_xgb   > threshold_buy  else
                     "Down" if r_xgb   < threshold_sell else "Neutral")
        sig_trans = ("Up"   if r_trans > threshold_buy  else
                     "Down" if r_trans < threshold_sell else "Neutral")
        is_buy = (sig_xgb == "Up" and sig_trans == "Up")

        cur_slip   = slippage_pct * 2.0 if vol_zscore[i] < -1.0 else slippage_pct
        buy_price  = df_test['open'].values[i] * (1 + cur_slip)
        sell_price = df_test['close'].values[i] * (1 - cur_slip)

        # Đóng vị thế cũ (sáng hôm sau)
        if position == 1:
            cash     = shares * sell_price * (1 - commission_pct)
            shares   = 0.0
            position = 0
            trades.append(dict(
                date=date_str(dates[i]), type="SELL",
                price=sell_price, cash=cash,
            ))

        # Mở vị thế mới (chiều hôm nay)
        if is_buy:
            shares   = cash * (1 - commission_pct) / buy_price
            cash     = 0.0
            position = 1
            trades.append(dict(
                date=date_str(dates[i]), type="BUY",
                price=buy_price, shares=shares,
            ))

        cur_equity = (
            cash if position == 0
            else shares * df_test['close'].values[i] * (1 - cur_slip) * (1 - commission_pct)
        )
        equity.append(cur_equity)
        bh_equity.append(bh_shares * df_test['close'].values[i])

    # Đóng vị thế cuối kỳ
    if position == 1:
        final_price = close_today[-1] * (1 - slippage_pct)
        cash = shares * final_price * (1 - commission_pct)
        trades.append(dict(
            date=date_str(dates[-1]), type="SELL",
            price=final_price, cash=cash,
        ))

    return dates, equity, bh_equity, trades


def compute_win_rate(trades, commission_pct):
    buy_trades  = [t for t in trades if t["type"] == "BUY"]
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    total = min(len(buy_trades), len(sell_trades))
    if total == 0:
        return 0.0, 0

    win = 0
    for b, s in zip(buy_trades[:total], sell_trades[:total]):
        # FIX: tính đủ commission cả 2 chiều
        buy_cost = b['price'] * b['shares'] * (1 + commission_pct)
        if s['cash'] > buy_cost:
            win += 1
    return win / total * 100, total


def run_walk_forward_evaluation(dates, equity, bh_equity):
    n = len(dates)
    w = n // 3
    results = []
    for i in range(3):
        s = i * w
        e = n if i == 2 else (i + 1) * w
        m  = compute_metrics(equity[s:e], bh_equity[s:e], dates[s:e])
        d0 = pd.to_datetime(dates[s]).strftime('%Y-%m-%d')
        d1 = pd.to_datetime(dates[e - 1]).strftime('%Y-%m-%d')
        results.append({"window": f"Window {i+1} ({d0} → {d1})", **m})
    return results


def main():
    TICKERS       = ["VNM.VN", "GOOGL", "META"]
    threshold_buy = 0.0010

    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
        elif arg == "ALL":
            pass
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
        print(f"\n{'='*70}\n📊 BACKTEST: {ticker}\n{'='*70}")

        commission_pct = 0.0020 if "VNM" in ticker.upper() else 0.0010
        slippage_pct   = 0.0010 if "VNM" in ticker.upper() else 0.0005

        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date="2026-05-20")

        dt = DataTransformer(time_steps=45)
        X_scaled, y_scaled, y_spread_scaled = dt.fit_transform_train_only(df, train_ratio=0.8)
        X_3D, y_3D, y_spread_3D = dt.create_sliding_windows(
            X_scaled, y_scaled, y_spread_scaled
        )

        _, _, X_test, y_test, y_test_raw, _, _ = \
            dt.split_train_test_chronological(
                df, X_3D, y_3D, y_spread_3D, train_ratio=0.8
            )

        # FIX: df_test chính xác + df_test_extended buffer +1 row
        df_align       = df.iloc[dt.time_steps:].reset_index(drop=True)
        split_idx      = int(len(X_3D) * 0.8)
        test_start_idx = split_idx + 45

        df_test = df_align.iloc[
            test_start_idx : test_start_idx + len(X_test)
        ].reset_index(drop=True)

        df_test_extended = df_align.iloc[
            test_start_idx : test_start_idx + len(X_test) + 1
        ].reset_index(drop=True)

        # Kiểm tra model tồn tại
        xgb_path   = os.path.join(models_dir, f'xgboost_model_{ticker}.pkl')
        trans_path = os.path.join(models_dir, f'transformer_model_{ticker}.keras')
        feat_path  = os.path.join(models_dir, f'feature_scaler_{ticker}.pkl')
        targ_path  = os.path.join(models_dir, f'target_scaler_{ticker}.pkl')

        if not (os.path.exists(xgb_path) and os.path.exists(trans_path)):
            print(f"❌ Không tìm thấy model cho {ticker}. Chạy run_training.py trước.")
            continue

        # Kiểm tra model có mới hơn config tuning không
        import datetime as dt_module
        trans_mtime = os.path.getmtime(trans_path)
        cfg_path    = os.path.join(
            ROOT_DIR, 'config', f'best_transformer_params_{ticker}.json'
        )
        if os.path.exists(cfg_path):
            cfg_mtime = os.path.getmtime(cfg_path)
            print(f"   📁 Model trained : "
                  f"{dt_module.datetime.fromtimestamp(trans_mtime).strftime('%Y-%m-%d %H:%M')}")
            print(f"   📁 Tuning config : "
                  f"{dt_module.datetime.fromtimestamp(cfg_mtime).strftime('%Y-%m-%d %H:%M')}")
            if cfg_mtime > trans_mtime:
                print(f"   ⚠️  Config tuning MỚI HƠN model! "
                      f"Cần chạy run_training.py trước.")
            else:
                print(f"   ✅ Model đã train SAU tuning — OK")

        xgb_model = joblib.load(xgb_path)
        from src.ai_models import load_multitask_model
        transformer_model = load_multitask_model(
            trans_path,
            input_shape=(45, X_test.shape[2]),
        )
        dt.feature_scaler = joblib.load(feat_path)
        dt.target_scaler  = joblib.load(targ_path)

        dates, equity, bh_equity, trades = run_simulation(
            df_test, df_test_extended,
            X_test, y_test_raw, dt,
            xgb_model, transformer_model, ticker,
            commission_pct, slippage_pct, threshold_buy, threshold_sell,
        )

        metrics              = compute_metrics(equity, bh_equity, dates)
        win_rate, total_lnhs = compute_win_rate(trades, commission_pct)

        print(f"\n🏆 KẾT QUẢ BACKTEST (Out-of-Sample):")
        print(f"   📈 Chiến lược : {metrics['strat_return']:+.2f}%")
        print(f"   📦 Buy&Hold  : {metrics['bh_return']:+.2f}%")
        print(f"   📊 Sharpe    : {metrics['sharpe']:.2f}")
        print(f"   📉 Max DD    : {metrics['mdd']:.2f}% (B&H: {metrics['bh_mdd']:.2f}%)")
        print(f"   🔔 Số lệnh   : {total_lnhs}")
        print(f"   🥇 Win Rate  : {win_rate:.2f}%")

        wf = run_walk_forward_evaluation(dates, equity, bh_equity)
        print(f"\n🎯 WALK-FORWARD (3 Windows):")
        print("-" * 95)
        print(f"{'Window':<40} | {'Strategy':<12} | {'B&H':<12} | {'Sharpe':<8} | {'MDD'}")
        print("-" * 95)
        for w in wf:
            print(f"{w['window']:<40} | {w['strat_return']:+11.2f}% | "
                  f"{w['bh_return']:+11.2f}% | {w['sharpe']:<8.2f} | {w['mdd']:.2f}%")
        print("-" * 95)

        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity,    label="Hybrid Strategy",
                 color='darkgreen', linewidth=2)
        plt.plot(dates, bh_equity, label="Buy & Hold",
                 color='grey', linestyle='--', alpha=0.8)
        plt.title(f"Equity Curve — {ticker}", fontsize=13, fontweight='bold')
        plt.xlabel("Ngày"); plt.ylabel("Tài sản (VNĐ)")
        plt.legend(); plt.grid(True, alpha=0.25)
        plot_path = os.path.join(
            figures_dir, f'backtest_equity_curve_{ticker}.png'
        )
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 {plot_path}")


if __name__ == "__main__":
    main()


