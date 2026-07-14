import numpy as np
import pandas as pd

# src/backtest_engine.py — Module dùng chung cho toàn bộ logic backtest.
# Được tách ra từ run_backtest.py, run_walk_forward_backtest.py, và
# run_training_transformer.py để loại bỏ trùng lặp code (~300 dòng giống nhau
# ở 3 nơi trước đây).
#
# QUAN TRỌNG: run_simulation() nhận final_returns — đây LUÔN LÀ kết quả
# Hybrid cuối cùng (sau khi đi qua XGBoost stacking), KHÔNG phải Transformer
# thô. Tên tham số được đổi từ "predicted_returns" sang "final_returns" để
# tránh nhầm lẫn với output trung gian của Transformer trong các pipeline
# có nhiều giai đoạn dự báo.


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
    final_returns,
    ticker, commission_pct, slippage_pct,
    threshold_buy=0.0010, threshold_sell=-0.0010,
    resume_threshold=0.10, cooldown_period=20,
    vol_ratio=None,
    vol_filter_threshold=None,
    hold_days=3,
):
    """
    Chạy mô phỏng giao dịch trên dữ liệu test.

    final_returns: kết quả dự báo CUỐI CÙNG (sau Hybrid XGBoost stacking),
    shape (N, 3) tương ứng [T+1, T+2, T+3]. Nếu shape (N, 1) hoặc (N,),
    tự động broadcast thành (N, 3) — hỗ trợ model cũ chỉ output 1 giá trị.

    df_test:          DataFrame test, length N (dùng cho close/volume/dates/regime).
    df_test_extended: DataFrame test có thêm 1 dòng buffer ở đầu (length N+1),
                       dùng để lấy open[0] làm base cho Buy & Hold.

    resume_threshold: % phục hồi của Buy & Hold (đo từ đáy kể từ lúc dừng lỗ)
                       cần đạt để tự động mở lại giao dịch (mặc định 10%).
                       Đo trên B&H thay vì trên equity chiến lược vì sau khi
                       dừng lỗ chiến lược chỉ còn cash, không phản ánh được
                       sự phục hồi của thị trường.
    cooldown_period:   Số phiên tối thiểu phải chờ kể từ lúc dừng lỗ trước khi
                        được phép mở lại giao dịch, kể cả khi đã đạt
                        resume_threshold sớm hơn. Tránh vòng lặp "cắt lỗ - hồi
                        giả 10% - vào lại - cắt lỗ tiếp" trong downtrend kéo
                        dài có nhiều bull trap (mặc định 20 phiên ≈ 1 tháng).
    """
    final_returns = np.asarray(final_returns)

    # Tự động mở rộng (N,) hoặc (N,1) → (N,3)
    if final_returns.ndim == 1:
        final_returns = final_returns.reshape(-1, 1)
    if final_returns.shape[1] == 1:
        final_returns = np.repeat(final_returns, 3, axis=1)

    # Sanity check — phát hiện scaler lỗi / inverse_transform sai
    assert np.abs(final_returns).max() < 0.20, \
        f"[SANITY] Final return bất thường: {np.abs(final_returns).max():.4f}"

    n = len(df_test)
    final_returns = final_returns[:n]

    # Debug correlation — phát hiện signal inversion
    close_vals = df_test['close'].values[:n]
    actual_returns = np.diff(close_vals) / (close_vals[:-1] + 1e-9)
    if len(actual_returns) > 1:
        corr = np.corrcoef(final_returns[:-1, 0], actual_returns)[0, 1]
        print(f"   [DEBUG] Corr Final-Actual (T+1)={corr:+.4f} | "
              f"Final mean={final_returns[:, 0].mean():.5f} | "
              f"Actual mean={actual_returns.mean():.5f}")

    dates = df_test['date'].values[:n]

    vol_series = df_test['volume'].values[:n]
    vol_mean   = pd.Series(vol_series).rolling(20, min_periods=1).mean().values
    vol_std    = pd.Series(vol_series).rolling(20, min_periods=1).std().fillna(1.0).values
    vol_zscore = (vol_series - vol_mean) / (vol_std + 1e-9)

    hist_returns = df_test['close'].pct_change().fillna(0.0).values[:n]
    rolling_std  = pd.Series(hist_returns).rolling(20, min_periods=1).std().fillna(0.02).values

    # Tính toán hoặc sử dụng vol_ratio truyền từ bên ngoài
    if vol_ratio is not None:
        vol_ratio_arr = np.asarray(vol_ratio)[:n]
    else:
        # Fallback tính toán trên df_test
        vol_250d = pd.Series(hist_returns).rolling(250, min_periods=1).std().fillna(0.02).values
        vol_ratio_arr = rolling_std / (vol_250d + 1e-9)

    initial_capital = 100_000_000.0
    cash     = initial_capital
    shares   = 0.0
    position = 0
    equity   = [initial_capital]

    # Buy & Hold: dùng close[0] làm base (nhất quán mark-to-market theo close)
    bh_shares = initial_capital / close_vals[0]
    bh_equity = [initial_capital]
    trades    = []

    date_str  = lambda d: d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
    buy_index = -1
    HOLD_DAYS = hold_days
    peak_price = 0.0
    max_equity_peak = initial_capital
    portfolio_stop_loss_triggered = False

    # Trạng thái cho cơ chế Resume sau Stop-Loss:
    # - bh_trough: đáy thấp nhất của B&H kể từ lần dừng lỗ GẦN NHẤT (reset
    #   mỗi lần trigger mới, không kế thừa từ lần trigger trước đó).
    # - stop_loss_trigger_index: chỉ số phiên (i) tại thời điểm trigger,
    #   dùng để tính cooldown_period (số phiên tối thiểu phải chờ).
    bh_trough = None
    stop_loss_trigger_index = None

    # Regime filter column theo ticker
    regime_col = 'sp500_above_ma200'
    has_regime = regime_col in df_test.columns
    if not has_regime:
        print(f"   [WARN] Không tìm thấy cột '{regime_col}' trong df_test — "
              f"Regime filter sẽ bị bỏ qua (không chặn mua).")

    open_vals  = df_test['open'].values[:n]
    close_vals_full = df_test['close'].values[:n]

    for i in range(n - 1):
        r_1 = final_returns[i, 0]
        r_2 = final_returns[i, 1]
        r_3 = final_returns[i, 2]
        sigma = rolling_std[i]

        # Sửa lỗi: Chỉ mua khi AI dự báo TĂNG.
        # (Logic cũ bắt mua khi AI dự báo sập r_1 < -1.5*sigma và đòi hỏi r_3 > r_2 > r_1 quá khắt khe)
        is_buy = (r_1 > threshold_buy)

        # Volatility Regime Filter
        if vol_filter_threshold is not None and vol_ratio_arr[i] > vol_filter_threshold:
            is_buy = False

        # Regime Filter
        if has_regime and df_test[regime_col].values[i] == 0:
            is_buy = False

        if portfolio_stop_loss_triggered:
            is_buy = False

        cur_slip   = slippage_pct * 2.0 if vol_zscore[i] < -1.0 else slippage_pct
        buy_price  = open_vals[i] * (1 + cur_slip)
        sell_price = close_vals_full[i] * (1 - cur_slip)

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
            current_close = close_vals_full[i]
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
            if is_buy:
                shares    = cash * (1 - commission_pct) / buy_price
                cash      = 0.0
                position  = 1
                buy_index = i
                peak_price = close_vals_full[i]
                trades.append(dict(
                    date=date_str(dates[i]), type="BUY",
                    price=buy_price, shares=shares,
                ))

        cur_equity = (
            cash if position == 0
            else shares * close_vals_full[i] * (1 - cur_slip) * (1 - commission_pct)
        )

        # Portfolio stop-loss -20%
        if cur_equity > max_equity_peak:
            max_equity_peak = cur_equity
        drawdown = (cur_equity - max_equity_peak) / max_equity_peak
        if drawdown <= -0.20 and not portfolio_stop_loss_triggered:
            portfolio_stop_loss_triggered = True
            stop_loss_trigger_index = i
            # Reset đáy B&H NGAY tại thời điểm trigger — không kế thừa đáy
            # từ lần trigger trước đó (nếu có), tránh dùng nhầm đáy cũ
            # (ví dụ đáy COVID 2020) làm tham chiếu cho lần dừng lỗ hiện tại.
            bh_trough = bh_equity[-1]
            currency_label = "USD"
            print(f"      🚨 [PORTFOLIO STOP-LOSS] Drawdown {drawdown*100:.2f}% "
                  f"vào {date_str(dates[i])} (Đỉnh: {max_equity_peak:,.2f} {currency_label}, "
                  f"Hiện tại: {cur_equity:,.2f} {currency_label}) — ngưng giao dịch.")
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

        # Resume sau Stop-Loss: chỉ xét khi đang bị khóa giao dịch.
        # Điều kiện mở lại (cả 2 đều phải thỏa):
        #   1. Đã qua cooldown_period phiên kể từ lúc trigger (tránh bull trap
        #      ngắn hạn ngay sau khi vừa cắt lỗ).
        #   2. B&H đã phục hồi >= resume_threshold từ đáy kể từ lúc trigger
        #      (đo trên B&H, không đo trên equity chiến lược, vì equity
        #      chiến lược lúc này chỉ là cash không đổi nên không phản ánh
        #      được sự phục hồi thực tế của thị trường).
        if portfolio_stop_loss_triggered:
            bh_trough = min(bh_trough, bh_equity[-1])
            recovery_from_trough = (bh_equity[-1] - bh_trough) / bh_trough
            sessions_since_trigger = i - stop_loss_trigger_index

            if (sessions_since_trigger >= cooldown_period
                    and recovery_from_trough >= resume_threshold):
                portfolio_stop_loss_triggered = False
                # Đặt lại đỉnh tài sản = vốn tiền mặt hiện tại, tránh việc
                # so sánh với đỉnh lịch sử cũ (trước khi sụp đổ) khiến hệ
                # thống bị kích hoạt dừng lỗ lại ngay lập tức.
                max_equity_peak = cur_equity
                bh_trough = None
                stop_loss_trigger_index = None
                print(f"      ✅ [RESUME] B&H phục hồi {recovery_from_trough*100:.1f}% "
                      f"từ đáy sau {sessions_since_trigger} phiên (cooldown "
                      f"{cooldown_period} phiên) — mở lại giao dịch tại "
                      f"{date_str(dates[i])}.")

        equity.append(cur_equity)
        bh_equity.append(bh_shares * close_vals_full[i])

    # Đóng vị thế cuối kỳ
    if position == 1:
        final_price = close_vals_full[-1] * (1 - slippage_pct)
        cash = shares * final_price * (1 - commission_pct)
        trades.append(dict(
            date=date_str(dates[-1]), type="SELL",
            price=final_price, cash=cash,
        ))

    return dates, equity, bh_equity, trades


def run_walk_forward_evaluation(dates, equity, bh_equity):
    """Chia equity curve thành 3 cửa sổ bằng nhau để đánh giá hiệu suất
    theo từng giai đoạn. dates length N, equity/bh_equity length N+1
    (có thêm initial_capital ở đầu) — index e dùng cho equity/bh_equity
    KHÔNG +1 vì compute_metrics tự lấy equity[-1]-equity[0]."""
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