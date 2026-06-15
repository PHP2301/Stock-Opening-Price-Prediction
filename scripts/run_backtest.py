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

    calmar = (
        (total_strat_ret / abs(mdd))
        if abs(mdd) > 1e-8
        else 0.0
    )

    return dict(
        strat_return=total_strat_ret,
        bh_return=total_bh_ret,
        sharpe=sharpe, mdd=mdd, bh_mdd=bh_mdd,
        calmar=calmar,
    )


def compute_trade_metrics(trades, commission_pct):
    buy_trades  = [t for t in trades if t["type"] == "BUY"]
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    total = min(len(buy_trades), len(sell_trades))
    if total == 0:
        return 0.0, 0, 1.0

    win = 0
    total_profits = 0.0
    total_losses = 0.0
    for b, s in zip(buy_trades[:total], sell_trades[:total]):
        # tính đủ commission cả 2 chiều
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
    win_rate = win / total * 100
    return win_rate, total, profit_factor


def run_simulation(
    df_test, df_test_extended,
    X_test, y_test_raw, dt,
    xgb_model, transformer_model,
    ticker, commission_pct, slippage_pct,
    threshold_buy=0.0010, threshold_sell=-0.0010,
    X_train=None, y_train=None,
):

    trans_pred_raw   = transformer_model.predict(X_test, verbose=0)
    trans_pred_clean = get_return_output(trans_pred_raw)
    trans_returns    = dt.target_scaler.inverse_transform(trans_pred_clean)

    xgb_returns = None
    if xgb_model is not None:
        print("   [PREDICT] Đang chạy dự báo trên tập test (sử dụng Rolling Retrain)...")
        xgb_returns = np.zeros((len(X_test), 3))
        X_test_today  = X_test[:, -1, :]
        X_test_hybrid = np.concatenate([trans_pred_clean, X_test_today], axis=1)
        xgb_pred_scaled = xgb_model.predict(X_test_hybrid)
        xgb_returns = dt.target_scaler.inverse_transform(xgb_pred_scaled)
    else:
        print("   [PREDICT] Đang chạy dự báo trên tập test (chỉ dùng Transformer)...")

    # Sanity check
    if xgb_returns is not None:
        assert np.abs(xgb_returns).max() < 0.20, \
            f"[SANITY] XGBoost return bất thường: {np.abs(xgb_returns).max():.4f}"
    assert np.abs(trans_returns).max() < 0.20, \
        f"[SANITY] Transformer return bất thường: {np.abs(trans_returns).max():.4f}"

    # Debug correlation
    actual_returns = np.diff(df_test['close'].values[:len(X_test)]) \
                     / (df_test['close'].values[:len(X_test) - 1] + 1e-9)
    corr_trans = np.corrcoef(trans_returns[:-1, 0], actual_returns)[0, 1]
    if xgb_returns is not None:
        corr_xgb   = np.corrcoef(xgb_returns[:-1, 0],  actual_returns)[0, 1]
        print(f"   [DEBUG] Corr XGB-Actual (T+1)={corr_xgb:+.4f} | "
              f"Corr Trans-Actual (T+1)={corr_trans:+.4f}")
    else:
        print(f"   [DEBUG] Corr Trans-Actual (T+1)={corr_trans:+.4f}")

    close_today = df_test['close'].values[:len(X_test)]
    dates       = df_test['date'].values[:len(X_test)]

    # FIX: open_tomorrow từ buffer row — không dùng shift(-1)+ffill
    open_tomorrow = df_test_extended['open'].values[1:len(X_test) + 1]

    # Volume zscore động
    vol_series = df_test['volume'].values[:len(X_test)]
    vol_mean   = pd.Series(vol_series).rolling(20, min_periods=1).mean().values
    vol_std    = pd.Series(vol_series).rolling(20, min_periods=1).std().fillna(1.0).values
    vol_zscore = (vol_series - vol_mean) / (vol_std + 1e-9)

    # Lợi nhuận thực tế lịch sử phục vụ tính toán dải Mean Reversion
    hist_returns = df_test['close'].pct_change().fillna(0.0).values
    rolling_std  = pd.Series(hist_returns).rolling(20, min_periods=1).std().fillna(0.02).values

    initial_capital = 100_000_000.0
    cash     = initial_capital
    shares   = 0.0
    position = 0
    equity    = [initial_capital]
    bh_shares = initial_capital / df_test['open'].values[0]
    bh_equity = [initial_capital]
    trades    = []

    date_str = lambda d: d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)

    buy_index = -1
    HOLD_DAYS = 3
    peak_price = 0.0

    max_equity_peak = initial_capital
    portfolio_stop_loss_triggered = False

    for i in range(len(X_test) - 1):
        # Rolling retrain cho XGBoost mỗi 126 phiên (6 tháng)
        if i > 0 and i % 126 == 0 and X_train is not None and y_train is not None and xgb_model is not None:
            print(f"      🔄 [ROLLING RETRAIN] Huấn luyện lại XGBoost tại ngày {date_str(dates[i])}...")
            # Trích xuất predictions cho phần test đã qua
            X_new_today  = X_test[:i, -1, :]
            X_new_hybrid = np.concatenate([trans_pred_clean[:i], X_new_today], axis=1)
            
            # Chuẩn hóa y_test_raw thành scaled target cho việc huấn luyện
            y_test_scaled = dt.target_scaler.transform(y_test_raw)
            y_new_xgb    = y_test_scaled[:i]

            # Trích xuất predictions cho tập train ban đầu
            X_train_today  = X_train[:, -1, :]
            trans_pred_train_raw = transformer_model.predict(X_train, verbose=0)
            trans_pred_train = get_return_output(trans_pred_train_raw)
            X_train_hybrid = np.concatenate([trans_pred_train, X_train_today], axis=1)

            # Gộp và huấn luyện
            X_expanded = np.concatenate([X_train_hybrid, X_new_hybrid], axis=0)
            y_expanded = np.concatenate([y_train, y_new_xgb], axis=0)

            mask = ~np.isnan(y_expanded).any(axis=1)
            X_expanded = X_expanded[mask]
            y_expanded = y_expanded[mask]

            from src.ai_models import build_xgboost_optimized
            xgb_model = build_xgboost_optimized(X_expanded, y_expanded)

            # Tính lại toàn bộ xgb_returns từ ngày i trở đi
            X_rem_today  = X_test[i:, -1, :]
            X_rem_hybrid = np.concatenate([trans_pred_clean[i:], X_rem_today], axis=1)
            xgb_pred_rem_scaled = xgb_model.predict(X_rem_hybrid)
            xgb_returns[i:] = dt.target_scaler.inverse_transform(xgb_pred_rem_scaled)

        # Tạo dự báo đồng thuận (ensemble) nếu có cả 2 mô hình
        if xgb_returns is not None:
            r_1 = 0.5 * trans_returns[i, 0] + 0.5 * xgb_returns[i, 0]
            r_2 = 0.5 * trans_returns[i, 1] + 0.5 * xgb_returns[i, 1]
            r_3 = 0.5 * trans_returns[i, 2] + 0.5 * xgb_returns[i, 2]
        else:
            r_1, r_2, r_3 = trans_returns[i, 0], trans_returns[i, 1], trans_returns[i, 2]

        sigma = rolling_std[i]

        # Tín hiệu Momentum: Dự báo tăng liên tục trong 3 ngày và ngày T+1 vượt ngưỡng mua
        is_momentum = (r_3 > r_2) and (r_2 > r_1) and (r_1 > threshold_buy)
        # Tín hiệu Mean Reversion: Dự báo ngày T+1 bị giảm cực mạnh, có khả năng bật tăng trở lại
        is_mean_rev  = (r_1 < -1.5 * sigma)

        is_buy = is_momentum or is_mean_rev

        if portfolio_stop_loss_triggered:
            is_buy = False

        cur_slip   = slippage_pct * 2.0 if vol_zscore[i] < -1.0 else slippage_pct
        buy_price  = df_test['open'].values[i] * (1 + cur_slip)
        sell_price = df_test['close'].values[i] * (1 - cur_slip)

        just_sold = False
        # Đóng vị thế cũ sau HOLD_DAYS ngày
        if position == 1 and (i - buy_index == HOLD_DAYS):
            cash     = shares * sell_price * (1 - commission_pct)
            shares   = 0.0
            position = 0
            buy_index = -1
            just_sold = True
            trades.append(dict(
                date=date_str(dates[i]), type="SELL",
                price=sell_price, cash=cash,
            ))

        # META G3: Trailing stop 4% từ đỉnh
        if position == 1 and not just_sold:
            current_close = df_test['close'].values[i]
            peak_price = max(peak_price, current_close)
            if ticker == "META" and current_close < peak_price * 0.96:
                cash     = shares * sell_price * (1 - commission_pct)
                shares   = 0.0
                position = 0
                buy_index = -1
                just_sold = True
                trades.append(dict(
                    date=date_str(dates[i]), type="SELL",
                    price=sell_price, cash=cash,
                    note="TRAILING_STOP"
                ))

        # Mở vị thế mới nếu chưa có vị thế và có tín hiệu mua, và không vừa bán ngày hôm nay
        if position == 0 and is_buy and not just_sold:
            # VNM G2: Bộ lọc thanh khoản
            if "VNM" in ticker.upper() and vol_zscore[i] < -0.5:
                is_buy = False

            if is_buy:
                shares   = cash * (1 - commission_pct) / buy_price
                cash     = 0.0
                position = 1
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

        # Giám sát portfolio stop-loss (-20% MDD)
        if cur_equity > max_equity_peak:
            max_equity_peak = cur_equity
            
        drawdown = (cur_equity - max_equity_peak) / max_equity_peak
        if drawdown <= -0.20 and not portfolio_stop_loss_triggered:
            portfolio_stop_loss_triggered = True
            print(f"      🚨 [PORTFOLIO STOP-LOSS] Drawdown chạm {drawdown*100:.2f}% vào ngày {date_str(dates[i])} (Đỉnh: {max_equity_peak:,.2f} VNĐ, Hiện tại: {cur_equity:,.2f} VNĐ). Kích hoạt dừng lỗ toàn bộ danh mục và ngưng giao dịch.")
            if position == 1:
                cash     = shares * sell_price * (1 - commission_pct)
                shares   = 0.0
                position = 0
                buy_index = -1
                trades.append(dict(
                    date=date_str(dates[i]), type="SELL",
                    price=sell_price, cash=cash,
                    note="PORTFOLIO_STOP_LOSS"
                ))
                cur_equity = cash

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
    trans_only = True
    args_cleaned = []
    for arg in sys.argv:
        if arg.lower() == "--xgb-hybrid":
            trans_only = False
        else:
            args_cleaned.append(arg)

    TICKERS       = ["VNM.VN", "GOOGL", "META"]
    threshold_buy = 0.0010

    if len(args_cleaned) > 1:
        arg = args_cleaned[1].upper()
        if arg in [t.upper() for t in TICKERS]:
            TICKERS = [t for t in TICKERS if t.upper() == arg]
        elif arg == "ALL":
            pass
    if len(args_cleaned) > 2:
        try:
            threshold_buy = float(args_cleaned[2])
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

        X_train, y_train, X_test, y_test, y_test_raw, _, _ = \
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

        if trans_only:
            if not os.path.exists(trans_path):
                print(f"❌ Không tìm thấy model Transformer cho {ticker}. Chạy run_training.py trước.")
                continue
        else:
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
                      f"Cần chạy huấn luyện trước.")
            else:
                print(f"   ✅ Model đã train SAU tuning — OK")

        xgb_model = None
        if not trans_only:
            xgb_model = joblib.load(xgb_path)
            
        from src.ai_models import PositionalEmbedding, TimeDecayAttention, UncertaintyWeightsLayer, MultiTaskModel
        custom_objects = {
            'PositionalEmbedding': PositionalEmbedding,
            'TimeDecayAttention': TimeDecayAttention,
            'UncertaintyWeightsLayer': UncertaintyWeightsLayer,
            'MultiTaskModel': MultiTaskModel,
        }
        transformer_model = tf.keras.models.load_model(
            trans_path,
            custom_objects=custom_objects,
            safe_mode=False
        )
        dt.feature_scaler = joblib.load(feat_path)
        dt.target_scaler  = joblib.load(targ_path)

        # Tỷ lệ threshold động cho từng ticker
        cur_threshold_buy = 0.0050 if "VNM" in ticker.upper() else 0.0010
        if len(args_cleaned) > 2:
            try:
                cur_threshold_buy = float(args_cleaned[2])
            except ValueError:
                pass
        cur_threshold_sell = -cur_threshold_buy

        dates, equity, bh_equity, trades = run_simulation(
            df_test, df_test_extended,
            X_test, y_test_raw, dt,
            xgb_model, transformer_model, ticker,
            commission_pct, slippage_pct, cur_threshold_buy, cur_threshold_sell,
            X_train, y_train
        )

        metrics              = compute_metrics(equity, bh_equity, dates)
        win_rate, total_lnhs, profit_factor = compute_trade_metrics(trades, commission_pct)

        print(f"\n🏆 KẾT QUẢ BACKTEST (Out-of-Sample):")
        print(f"   📈 Chiến lược : {metrics['strat_return']:+.2f}%")
        print(f"   📦 Buy&Hold  : {metrics['bh_return']:+.2f}%")
        print(f"   {ticker} Static: Strategy {metrics['strat_return']:+.2f}% vs B&H {metrics['bh_return']:+.2f}%")
        print(f"   📊 Sharpe    : {metrics['sharpe']:.2f}")
        print(f"   📉 Max DD    : {metrics['mdd']:.2f}% (B&H: {metrics['bh_mdd']:.2f}%)")
        print(f"   ⚖️  Calmar    : {metrics['calmar']:.2f}")
        print(f"   🔔 Số lệnh   : {total_lnhs}")
        print(f"   🥇 Win Rate  : {win_rate:.2f}%")
        print(f"   💵 Profit F.  : {profit_factor:.2f}")

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

        # Lưu hiệu suất backtest vào JSON
        config_dir = os.path.join(ROOT_DIR, 'config')
        os.makedirs(config_dir, exist_ok=True)
        perf_path = os.path.join(config_dir, f'performance_metrics_{ticker}.json')
        
        # Tính drawdown cuối kỳ của backtest làm proxy cho current_drawdown
        current_drawdown = (equity[-1] - max(equity)) / max(equity) if len(equity) > 0 else 0.0

        perf_data = {
            'overall_win_rate': win_rate / 100.0,
            'total_trades': total_lnhs,
            'profit_factor': profit_factor,
            'strat_return': metrics['strat_return'],
            'bh_return': metrics['bh_return'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['mdd'],
            'current_drawdown': current_drawdown,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(perf_data, f, indent=4)
        print(f"💾 Đã lưu hiệu suất backtest vào: {perf_path}")


if __name__ == "__main__":
    main()


