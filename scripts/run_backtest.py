import os, sys, datetime, random, joblib, json
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

# run_backtest.py — Static 80/20 split, Transformer thuần (không XGBoost)
# Các fixes so với bản cũ:
# 1. Xóa num_features logic → DataTransformer(time_steps=45) không tham số thừa
# 2. FIX: bh_equity dùng close[0] làm base (nhất quán với mark-to-market theo close)
# 3. FIX: trans_returns shape check — nếu model cũ output (N,1) thay vì (N,3)
#    sẽ broadcast đúng thay vì IndexError tại index 1,2
# 4. Threshold default per-ticker tách riêng khỏi tham số dòng lệnh để rõ ràng


ALERT_THRESHOLD = {
    'VNM.VN': 0.03,   # 3.0% — thị trường VN ít biến động
    'GOOGL':  0.025,  # 2.5%
    'META':   0.025,  # 2.5%
}


def get_return_output(pred):
    """Handle Transformer output: list [return, spread] hoặc tensor đơn."""
    return pred[0] if isinstance(pred, (list, tuple)) else pred


def compute_metrics(equity, bh_equity, dates):
    equity    = np.array(equity)
    bh_equity = np.array(bh_equity)
    if len(equity) < 2 or len(bh_equity) < 2:
        return dict(strat_return=0.0, bh_return=0.0,
                    sharpe=0.0, mdd=0.0, bh_mdd=0.0, calmar=0.0)

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
    calmar   = (total_strat_ret / abs(mdd)) if abs(mdd) > 1e-8 else 0.0

    return dict(
        strat_return=total_strat_ret, bh_return=total_bh_ret,
        sharpe=sharpe, mdd=mdd, bh_mdd=bh_mdd, calmar=calmar,
    )


def compute_trade_metrics(trades, commission_pct):
    buy_trades  = [t for t in trades if t["type"] == "BUY"]
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    total = min(len(buy_trades), len(sell_trades))
    if total == 0:
        return 0.0, 0, 1.0

    win = 0
    total_profits = 0.0
    total_losses  = 0.0
    for b, s in zip(buy_trades[:total], sell_trades[:total]):
        buy_cost = b['price'] * b['shares'] * (1 + commission_pct)
        pnl = s['cash'] - buy_cost
        if pnl > 0:
            win += 1
            total_profits += pnl
        else:
            total_losses += abs(pnl)

    profit_factor = (
        total_profits / total_losses
        if total_losses > 1e-8
        else (999.0 if total_profits > 0 else 1.0)
    )
    return win / total * 100, total, profit_factor


def run_simulation(
    df_test, df_test_extended,
    X_test, dt, transformer_model,
    ticker, commission_pct, slippage_pct,
    threshold_buy=0.0010, threshold_sell=-0.0010,
):
    # ── Dự báo ────────────────────────────────────────────────────────
    trans_pred_raw   = transformer_model.predict(X_test, verbose=0)
    trans_pred_clean = get_return_output(trans_pred_raw)
    trans_returns    = dt.target_scaler.inverse_transform(trans_pred_clean)

    # FIX 3: nếu model cũ output shape (N,1), expand thành (N,3) để
    # tránh IndexError khi lấy trans_returns[i, 1] và [i, 2]
    if trans_returns.ndim == 1:
        trans_returns = trans_returns.reshape(-1, 1)
    if trans_returns.shape[1] == 1:
        trans_returns = np.repeat(trans_returns, 3, axis=1)

    # Sanity check
    assert np.abs(trans_returns).max() < 0.20, \
        f"[SANITY] Transformer return bất thường: {np.abs(trans_returns).max():.4f}"

    # Debug correlation — phát hiện signal inversion
    actual_returns = (
        np.diff(df_test['close'].values[:len(X_test)])
        / (df_test['close'].values[:len(X_test) - 1] + 1e-9)
    )
    corr_trans = np.corrcoef(trans_returns[:-1, 0], actual_returns)[0, 1]
    print(f"   [DEBUG] Corr Trans-Actual (T+1)={corr_trans:+.4f} | "
          f"Trans mean={trans_returns[:, 0].mean():.5f} | "
          f"Actual mean={actual_returns.mean():.5f}")

    # ── Chuẩn bị mảng dữ liệu ─────────────────────────────────────────
    close_today   = df_test['close'].values[:len(X_test)]
    open_tomorrow = df_test_extended['open'].values[1:len(X_test) + 1]
    dates         = df_test['date'].values[:len(X_test)]

    vol_series = df_test['volume'].values[:len(X_test)]
    vol_mean   = pd.Series(vol_series).rolling(20, min_periods=1).mean().values
    vol_std    = pd.Series(vol_series).rolling(20, min_periods=1).std().fillna(1.0).values
    vol_zscore = (vol_series - vol_mean) / (vol_std + 1e-9)

    hist_returns = df_test['close'].pct_change().fillna(0.0).values
    rolling_std  = pd.Series(hist_returns).rolling(20, min_periods=1).std().fillna(0.02).values

    # ── Khởi tạo simulation ───────────────────────────────────────────
    initial_capital = 100_000_000.0
    cash      = initial_capital
    shares    = 0.0
    position  = 0
    equity    = [initial_capital]

    # FIX 2: bh_shares dùng close[0] làm base (nhất quán mark-to-market theo close)
    bh_shares = initial_capital / close_today[0]
    bh_equity = [initial_capital]
    trades    = []

    date_str  = lambda d: d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
    buy_index = -1
    HOLD_DAYS = 3
    peak_price = 0.0
    max_equity_peak = initial_capital
    portfolio_stop_loss_triggered = False

    # Regime filter column theo ticker
    regime_col = 'vnm_etf_above_ma200' if "VNM" in ticker.upper() else 'sp500_above_ma200'
    has_regime = regime_col in df_test.columns
    if not has_regime:
        print(f"   [WARN] Không tìm thấy cột '{regime_col}' trong df_test — "
              f"Regime filter sẽ bị bỏ qua (không chặn mua).")

    # ── Vòng lặp giao dịch ────────────────────────────────────────────
    for i in range(len(X_test) - 1):
        r_1 = trans_returns[i, 0]
        r_2 = trans_returns[i, 1]
        r_3 = trans_returns[i, 2]
        sigma = rolling_std[i]

        is_momentum = (r_3 > r_2) and (r_2 > r_1) and (r_1 > threshold_buy)
        is_mean_rev = (r_1 < -1.5 * sigma)
        is_buy = is_momentum or is_mean_rev

        # Regime Filter
        if has_regime and df_test[regime_col].values[i] == 0:
            is_buy = False

        if portfolio_stop_loss_triggered:
            is_buy = False

        cur_slip   = slippage_pct * 2.0 if vol_zscore[i] < -1.0 else slippage_pct
        buy_price  = df_test['open'].values[i] * (1 + cur_slip)
        sell_price = df_test['close'].values[i] * (1 - cur_slip)

        just_sold = False

        # Đóng vị thế sau HOLD_DAYS
        if position == 1 and (i - buy_index == HOLD_DAYS):
            cash      = shares * sell_price * (1 - commission_pct)
            shares    = 0.0
            position  = 0
            buy_index = -1
            just_sold = True
            trades.append(dict(
                date=date_str(dates[i]), type="SELL",
                price=sell_price, cash=cash,
            ))

        # Trailing stop 4% (META)
        if position == 1 and not just_sold:
            current_close = df_test['close'].values[i]
            peak_price = max(peak_price, current_close)
            if ticker == "META" and current_close < peak_price * 0.96:
                cash      = shares * sell_price * (1 - commission_pct)
                shares    = 0.0
                position  = 0
                buy_index = -1
                just_sold = True
                trades.append(dict(
                    date=date_str(dates[i]), type="SELL",
                    price=sell_price, cash=cash,
                    note="TRAILING_STOP",
                ))

        # Mở vị thế mới
        if position == 0 and is_buy and not just_sold:
            if "VNM" in ticker.upper() and vol_zscore[i] < -0.5:
                is_buy = False
            if is_buy:
                shares    = cash * (1 - commission_pct) / buy_price
                cash      = 0.0
                position  = 1
                buy_index = i
                peak_price = df_test['close'].values[i]
                trades.append(dict(
                    date=date_str(dates[i]), type="BUY",
                    price=buy_price, shares=shares,
                ))

        cur_equity = (
            cash if position == 0
            else shares * df_test['close'].values[i] * (1 - cur_slip) * (1 - commission_pct)
        )

        # Portfolio stop-loss -20%
        if cur_equity > max_equity_peak:
            max_equity_peak = cur_equity
        drawdown = (cur_equity - max_equity_peak) / max_equity_peak
        if drawdown <= -0.20 and not portfolio_stop_loss_triggered:
            portfolio_stop_loss_triggered = True
            print(f"      🚨 [PORTFOLIO STOP-LOSS] Drawdown {drawdown*100:.2f}% "
                  f"vào {date_str(dates[i])} — ngưng giao dịch.")
            if position == 1:
                cash      = shares * sell_price * (1 - commission_pct)
                shares    = 0.0
                position  = 0
                buy_index = -1
                trades.append(dict(
                    date=date_str(dates[i]), type="SELL",
                    price=sell_price, cash=cash,
                    note="PORTFOLIO_STOP_LOSS",
                ))
                cur_equity = cash

        equity.append(cur_equity)
        bh_equity.append(bh_shares * close_today[i])

    # Đóng vị thế cuối kỳ
    if position == 1:
        final_price = close_today[-1] * (1 - slippage_pct)
        cash = shares * final_price * (1 - commission_pct)
        trades.append(dict(
            date=date_str(dates[-1]), type="SELL",
            price=final_price, cash=cash,
        ))

    return dates, equity, bh_equity, trades


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
    args_cleaned  = sys.argv
    TICKERS       = ["VNM.VN", "GOOGL", "META"]
    cli_threshold = None   # threshold từ dòng lệnh (override per-ticker default)

    if len(args_cleaned) > 1:
        arg = args_cleaned[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
        elif arg == "ALL":
            pass
    if len(args_cleaned) > 2:
        try:
            cli_threshold = float(args_cleaned[2])
        except ValueError:
            pass

    models_dir  = os.path.join(ROOT_DIR, 'models')
    figures_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    config_dir  = os.path.join(ROOT_DIR, 'config')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(config_dir,  exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'='*70}\n📊 BACKTEST: {ticker}\n{'='*70}")

        commission_pct = 0.0020 if "VNM" in ticker.upper() else 0.0010
        slippage_pct   = 0.0010 if "VNM" in ticker.upper() else 0.0005

        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date="2026-05-20")

        dt = DataTransformer(time_steps=45, num_features=42)
        X_scaled, y_scaled, y_spread_scaled = dt.fit_transform_train_only(df, train_ratio=0.8)
        X_3D, y_3D, y_spread_3D = dt.create_sliding_windows(
            X_scaled, y_scaled, y_spread_scaled
        )

        _, _, X_test, _, _, _, _ = dt.split_train_test_chronological(
            df, X_3D, y_3D, y_spread_3D, train_ratio=0.8
        )

        df_align       = df.iloc[dt.time_steps:].reset_index(drop=True)
        split_idx      = int(len(X_3D) * 0.8)
        test_start_idx = split_idx + 45

        df_test = df_align.iloc[
            test_start_idx : test_start_idx + len(X_test)
        ].reset_index(drop=True)

        df_test_extended = df_align.iloc[
            test_start_idx : test_start_idx + len(X_test) + 1
        ].reset_index(drop=True)

        # Kiểm tra model tồn tại trước khi tải dữ liệu (fail-fast)
        trans_path = os.path.join(models_dir, f'transformer_model_{ticker}.keras')
        feat_path  = os.path.join(models_dir, f'feature_scaler_{ticker}.pkl')
        targ_path  = os.path.join(models_dir, f'target_scaler_{ticker}.pkl')

        # Fallback tìm model có timestamp gần nhất nếu file gốc không tồn tại
        if not os.path.exists(trans_path):
            import glob
            timestamped_models = glob.glob(os.path.join(models_dir, f'transformer_model_{ticker}_*.keras'))
            if timestamped_models:
                timestamped_models.sort(key=os.path.getmtime, reverse=True)
                trans_path = timestamped_models[0]
                print(f"   ℹ️ Không thấy model gốc, tự động sử dụng model timestamp mới nhất: {os.path.basename(trans_path)}")
            else:
                print(f"❌ Không tìm thấy model Transformer cho {ticker}. Chạy run_training_transformer.py trước.")
                continue

        if not os.path.exists(feat_path) or not os.path.exists(targ_path):
            print(f"❌ Không tìm thấy feature/target scaler cho {ticker}. Chạy run_training_transformer.py trước.")
            continue

        # Kiểm tra timestamp model vs tuning config
        import datetime as dt_module
        trans_mtime = os.path.getmtime(trans_path)
        cfg_path    = os.path.join(config_dir, f'best_transformer_params_{ticker}.json')
        if os.path.exists(cfg_path):
            cfg_mtime = os.path.getmtime(cfg_path)
            print(f"   📁 Model trained : "
                  f"{dt_module.datetime.fromtimestamp(trans_mtime).strftime('%Y-%m-%d %H:%M')}")
            print(f"   📁 Tuning config : "
                  f"{dt_module.datetime.fromtimestamp(cfg_mtime).strftime('%Y-%m-%d %H:%M')}")
            if cfg_mtime > trans_mtime:
                print(f"   ⚠️  Config tuning MỚI HƠN model — cần retrain trước khi backtest.")
            else:
                print(f"   ✅ Model đã train SAU tuning — OK")

        custom_objects = {
            'PositionalEmbedding':     PositionalEmbedding,
            'TimeDecayAttention':      TimeDecayAttention,
            'UncertaintyWeightsLayer': UncertaintyWeightsLayer,
            'MultiTaskModel':          MultiTaskModel,
        }
        transformer_model = tf.keras.models.load_model(
            trans_path, custom_objects=custom_objects, safe_mode=False
        )
        dt.feature_scaler = joblib.load(feat_path)
        dt.target_scaler  = joblib.load(targ_path)

        # Threshold: dùng CLI override nếu có, ngược lại dùng per-ticker default
        if cli_threshold is not None:
            cur_threshold_buy = cli_threshold
        else:
            cur_threshold_buy = 0.0050 if "VNM" in ticker.upper() else 0.0010
        cur_threshold_sell = -cur_threshold_buy
        print(f"   🎯 Threshold buy={cur_threshold_buy*100:.2f}% | sell={cur_threshold_sell*100:.2f}%")

        dates, equity, bh_equity, trades = run_simulation(
            df_test, df_test_extended,
            X_test, dt, transformer_model, ticker,
            commission_pct, slippage_pct,
            cur_threshold_buy, cur_threshold_sell,
        )

        metrics                          = compute_metrics(equity, bh_equity, dates)
        win_rate, total_lnhs, pf         = compute_trade_metrics(trades, commission_pct)

        # Debug cảnh báo biến động Telegram
        alert_thr = ALERT_THRESHOLD.get(ticker, 0.03)
        last_atr_pct = (
            (df_test['high'].iloc[-14:].values - df_test['low'].iloc[-14:].values)
            / df_test['close'].iloc[-14:].values
        ).mean() * 100 if len(df_test) >= 14 else 0.0
        print(f"   🔍 [Telegram check] ATR={last_atr_pct:.2f}% (ngưỡng {alert_thr*100:.1f}%) — "
              f"{'🚨 CẢNH BÁO' if last_atr_pct >= alert_thr*100 else '✅ Bình thường'}")

        print(f"\n🏆 KẾT QUẢ BACKTEST (Out-of-Sample):")
        print(f"   📈 Chiến lược : {metrics['strat_return']:+.2f}%")
        print(f"   📦 Buy&Hold  : {metrics['bh_return']:+.2f}%")
        print(f"   📊 Sharpe    : {metrics['sharpe']:.2f}")
        print(f"   📉 Max DD    : {metrics['mdd']:.2f}% (B&H: {metrics['bh_mdd']:.2f}%)")
        print(f"   ⚖️  Calmar    : {metrics['calmar']:.2f}")
        print(f"   🔔 Số lệnh   : {total_lnhs}")
        print(f"   🥇 Win Rate  : {win_rate:.2f}%")
        print(f"   💵 Profit F.  : {pf:.2f}")

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
        plt.plot(dates, equity,    label="Transformer Strategy", color='darkgreen', linewidth=2)
        plt.plot(dates, bh_equity, label="Buy & Hold",           color='grey', linestyle='--', alpha=0.8)
        plt.title(f"Equity Curve — {ticker}", fontsize=13, fontweight='bold')
        currency_label = "VNĐ" if "VNM" in ticker.upper() else "USD"
        plt.xlabel("Ngày"); plt.ylabel(f"Tài sản ({currency_label})")
        plt.legend(); plt.grid(True, alpha=0.25)
        plot_path = os.path.join(figures_dir, f'backtest_equity_curve_{ticker}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 {plot_path}")

        # Lưu hiệu suất vào JSON (dùng cho Kelly Criterion ở predict.py)
        perf_path = os.path.join(config_dir, f'performance_metrics_{ticker}.json')
        current_drawdown = (equity[-1] - max(equity)) / max(equity) if equity else 0.0
        perf_data = {
            'overall_win_rate':  win_rate / 100.0,
            'total_trades':      total_lnhs,
            'profit_factor':     pf,
            'strat_return':      metrics['strat_return'],
            'bh_return':         metrics['bh_return'],
            'sharpe':            metrics['sharpe'],
            'max_drawdown':      metrics['mdd'],
            'current_drawdown':  current_drawdown,
            'timestamp':         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(perf_data, f, indent=4)
        print(f"💾 Hiệu suất → {perf_path}")


if __name__ == "__main__":
    main()