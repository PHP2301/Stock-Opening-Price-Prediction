import os, sys, datetime, random, joblib, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data, format_vn, get_realtime_usd_vnd_rate
from src.features import DataTransformer
from src.ai_models import (
    PositionalEmbedding, TimeDecayAttention,
    MultiTaskModel, UncertaintyWeightsLayer, build_transformer, build_xgboost_optimized
)

USD_TO_VND = get_realtime_usd_vnd_rate()


def get_return_output(pred):
    """Lấy đầu ra return của Transformer."""
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
    xgb_returns, trans_returns,
    ticker, commission_pct, slippage_pct,
    threshold_buy=0.0010, threshold_sell=-0.0010,
):
    vol_series = df_test['volume'].values
    vol_mean   = pd.Series(vol_series).rolling(20, min_periods=1).mean().values
    vol_std    = pd.Series(vol_series).rolling(20, min_periods=1).std().fillna(1.0).values
    vol_zscore = (vol_series - vol_mean) / (vol_std + 1e-9)

    hist_returns = df_test['close'].pct_change().fillna(0.0).values
    rolling_std  = pd.Series(hist_returns).rolling(20, min_periods=1).std().fillna(0.02).values

    initial_capital = 100_000_000.0
    cash     = initial_capital
    shares   = 0.0
    position = 0
    equity    = [initial_capital]
    bh_shares = initial_capital / df_test_extended['open'].values[0]
    bh_equity = [initial_capital]
    trades    = []

    date_str = lambda d: d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
    dates = df_test['date'].values

    buy_index = -1
    HOLD_DAYS = 3

    max_equity_peak = initial_capital
    portfolio_stop_loss_triggered = False

    for i in range(len(df_test) - 1):
        if xgb_returns is not None:
            r_1 = 0.5 * trans_returns[i, 0] + 0.5 * xgb_returns[i, 0]
            r_2 = 0.5 * trans_returns[i, 1] + 0.5 * xgb_returns[i, 1]
            r_3 = 0.5 * trans_returns[i, 2] + 0.5 * xgb_returns[i, 2]
        else:
            r_1, r_2, r_3 = trans_returns[i, 0], trans_returns[i, 1], trans_returns[i, 2]

        sigma = rolling_std[i]

        is_momentum = (r_3 > r_2) and (r_2 > r_1) and (r_1 > threshold_buy)
        is_mean_rev  = (r_1 < -1.5 * sigma)
        is_buy = is_momentum or is_mean_rev

        if portfolio_stop_loss_triggered:
            is_buy = False

        cur_slip   = slippage_pct * 2.0 if vol_zscore[i] < -1.0 else slippage_pct
        buy_price  = df_test_extended['open'].values[i] * (1 + cur_slip)
        sell_price = df_test_extended['close'].values[i] * (1 - cur_slip)

        just_sold = False
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

        if position == 0 and is_buy and not just_sold:
            shares   = cash * (1 - commission_pct) / buy_price
            cash     = 0.0
            position = 1
            buy_index = i
            trades.append(dict(
                date=date_str(dates[i]), type="BUY",
                price=buy_price, shares=shares,
            ))

        cur_equity = (
            cash if position == 0
            else shares * df_test_extended['close'].values[i] * (1 - cur_slip) * (1 - commission_pct)
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
        bh_equity.append(bh_shares * df_test_extended['close'].values[i])

    if position == 1:
        final_price = df_test_extended['close'].values[-1] * (1 - slippage_pct)
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
        m  = compute_metrics(equity[s:e+1], bh_equity[s:e+1], dates[s:e])
        d0 = pd.to_datetime(dates[s]).strftime('%Y-%m-%d')
        d1 = pd.to_datetime(dates[e - 1]).strftime('%Y-%m-%d')
        results.append({"window": f"Window {i+1} ({d0} → {d1})", **m})
    return results


def main():
    TICKERS       = ["VNM.VN", "GOOGL", "META"]
    threshold_buy = 0.0010

    # Parse arguments
    args_cleaned = [arg for arg in sys.argv if not arg.startswith("--")]
    trans_only   = "--trans-only" in sys.argv

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
    figures_dir    = os.path.join(ROOT_DIR, 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    test_years = [2023, 2024, 2025, 2026]

    for ticker in TICKERS:
        print(f"\n{'='*80}")
        print(f"🔄 BẮT ĐẦU ROLLING WALK-FORWARD BACKTEST: {ticker}")
        print(f"{'='*80}")

        commission_pct = 0.0020 if "VNM" in ticker.upper() else 0.0010
        slippage_pct   = 0.0010 if "VNM" in ticker.upper() else 0.0005

        # Nạp toàn bộ dữ liệu từ 2012
        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date="2026-05-20")
        df['date'] = pd.to_datetime(df['date'])

        # Tìm chỉ mục bắt đầu kiểm thử thực tế (ngày giao dịch đầu tiên của năm 2023)
        test_start_date = pd.to_datetime("2023-01-01")
        if df['date'].max() < test_start_date:
            print(f"❌ Dữ liệu mã {ticker} quá ngắn, không có dữ liệu sau 2023-01-01!")
            continue

        test_start_idx = df[df['date'] >= test_start_date].index[0]
        N_test_total = len(df) - test_start_idx

        df_test_all = df.iloc[test_start_idx:].reset_index(drop=True)
        df_test_extended_all = df.iloc[test_start_idx : test_start_idx + N_test_total + 1].reset_index(drop=True)

        # Đọc tham số tối ưu từ config
        d_model, num_heads, dropout_rate, learning_rate, batch_size = 128, 8, 0.3, 1e-4, 64
        cfg_path = os.path.join(ROOT_DIR, 'config', f'best_transformer_params_{ticker}.json')
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    p = json.load(f)
                d_model       = p.get('d_model', d_model)
                num_heads     = p.get('num_heads', p.get('heads', num_heads))
                dropout_rate  = p.get('dropout_rate', dropout_rate)
                learning_rate = p.get('learning_rate', learning_rate)
                batch_size    = p.get('batch_size', batch_size)
                print(f"🥇 Đọc cấu hình tối ưu: d_model={d_model}, heads={num_heads}, batch_size={batch_size}")
            except Exception as e:
                print(f"⚠️ Lỗi đọc file config: {e}")

        # Danh sách lưu kết quả dự báo out-of-sample gộp qua các năm
        trans_pred_accum = []
        xgb_pred_accum   = []

        previous_weights = None

        for year in test_years:
            year_start = pd.to_datetime(f"{year}-01-01")
            year_end   = pd.to_datetime(f"{year}-12-31")

            # train_df: toàn bộ dữ liệu trước năm hiện tại
            train_df = df[df['date'] < year_start].copy()
            # test_df: dữ liệu của năm hiện tại
            test_df  = df[(df['date'] >= year_start) & (df['date'] <= year_end)].copy()

            if test_df.empty:
                print(f"   [INFO] Năm {year}: Không có dữ liệu kiểm thử, bỏ qua.")
                continue

            print(f"\n▶️ [WINDOW {year}] — Huấn luyện: 2012 → {year-1} ({len(train_df)} phiên) | Kiểm thử: {year} ({len(test_df)} phiên)...")

            # Khởi tạo DataTransformer riêng cho cửa sổ này để cô lập scaling hoàn toàn
            dt = DataTransformer(time_steps=45)

            # Fit scaler trên Train và transform Train
            X_train_raw = dt.transform_df(train_df).values
            y_train_raw = train_df[dt.target_cols].values
            
            dt.feature_scaler.fit(X_train_raw)
            dt.target_scaler.fit(y_train_raw)

            if all(col in train_df.columns for col in dt.spread_cols):
                y_train_spread_raw = train_df[dt.spread_cols].values
                dt.spread_scaler.fit(y_train_spread_raw)
                y_train_spread_scaled = dt.spread_scaler.transform(y_train_spread_raw)
            else:
                y_train_spread_scaled = None

            X_train_scaled = dt.feature_scaler.transform(X_train_raw)
            y_train_scaled = dt.target_scaler.transform(y_train_raw)

            X_train_3D, y_train_3D, y_train_spread_3D = dt.create_sliding_windows(
                X_train_scaled, y_train_scaled, y_train_spread_scaled
            )

            # Tách tập validation cho Transformer
            val_size = int(len(X_train_3D) * 0.1)
            purge = 45
            if val_size > 0 and len(X_train_3D) - val_size - purge > 0:
                train_end = len(X_train_3D) - val_size - purge
                X_tr, y_tr = X_train_3D[:train_end], y_train_3D[:train_end]
                X_va, y_va = X_train_3D[-val_size:], y_train_3D[-val_size:]
                y_tr_spread = y_train_spread_3D[:train_end] if y_train_spread_3D is not None else None
                y_va_spread = y_train_spread_3D[-val_size:] if y_train_spread_3D is not None else None
            else:
                X_tr, y_tr = X_train_3D, y_train_3D
                X_va, y_va = X_train_3D, y_train_3D
                y_tr_spread = y_train_spread_3D
                y_va_spread = y_train_spread_3D

            # ── BƯỚC 1: Xây dựng & Huấn luyện Transformer ─────────────────────
            transformer_model = build_transformer(
                input_shape=(X_train_3D.shape[1], X_train_3D.shape[2]),
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )

            if previous_weights is not None:
                print(f"      [WARM-START] Khởi tạo trọng số từ Window trước để tối ưu hóa học...")
                try:
                    transformer_model.set_weights(previous_weights)
                    epochs_to_run = 25
                    patience_run  = 8
                    lr_decay = 2e-5
                except Exception as e:
                    print(f"      ⚠️ Không load được trọng số: {e}, huấn luyện từ đầu.")
                    epochs_to_run = 100
                    patience_run  = 15
                    lr_decay = learning_rate
            else:
                epochs_to_run = 100
                patience_run  = 15
                lr_decay = learning_rate

            callbacks_window = [
                EarlyStopping(monitor='val_loss', patience=patience_run, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
            ]

            transformer_model.fit(
                X_tr,
                {"output_return": y_tr, "output_spread": y_tr_spread},
                validation_data=(X_va, {"output_return": y_va, "output_spread": y_va_spread}),
                epochs=epochs_to_run, batch_size=batch_size, callbacks=callbacks_window, verbose=0
            )

            # Lưu lại trọng số của window này cho window sau
            previous_weights = transformer_model.get_weights()

            feature_extractor = Model(
                inputs=transformer_model.inputs,
                outputs=transformer_model.get_layer("latent_embedding").output,
            )

            # ── BƯỚC 2: Trích latent features & Huấn luyện XGBoost ──────────────────
            xgb_model = None
            if not trans_only:
                X_train_latent = feature_extractor.predict(X_train_3D, verbose=0)
                X_train_today  = X_train_3D[:, -1, :]
                X_train_hybrid = np.concatenate([X_train_latent, X_train_today], axis=1)

                print(f"      🌳 [TRAIN] Huấn luyện Hybrid XGBoost ({X_train_hybrid.shape[1]}-dim)...")
                xgb_model = build_xgboost_optimized(X_train_hybrid, y_train_3D)

            # ── BƯỚC 3: Dự báo out-of-sample cho tập Test năm hiện tại ───────────
            # Ghép 45 ngày trễ từ cuối Train để cung cấp lookback đầy đủ cho các ngày đầu Test
            test_extended_df = pd.concat([train_df.tail(45), test_df], ignore_index=True)
            X_test_raw = dt.transform_df(test_extended_df).values
            y_test_raw_full = test_extended_df[dt.target_cols].values

            # Transform Test bằng Scaler đã fit hoàn toàn trên Train (không leak)
            X_test_scaled = dt.feature_scaler.transform(X_test_raw)
            y_test_scaled = dt.target_scaler.transform(y_test_raw_full)

            if all(col in test_extended_df.columns for col in dt.spread_cols):
                y_test_spread_raw = test_extended_df[dt.spread_cols].values
                y_test_spread_scaled = dt.spread_scaler.transform(y_test_spread_raw)
            else:
                y_test_spread_scaled = None

            X_test_3D, y_test_3D, y_test_spread_3D = dt.create_sliding_windows(
                X_test_scaled, y_test_scaled, y_test_spread_scaled
            )

            # Predict Transformer
            trans_pred_raw   = transformer_model.predict(X_test_3D, verbose=0)
            trans_pred_clean = get_return_output(trans_pred_raw)
            trans_pred_ret   = dt.target_scaler.inverse_transform(trans_pred_clean)
            trans_pred_accum.append(trans_pred_ret)

            # Predict XGBoost
            if not trans_only:
                X_test_latent = feature_extractor.predict(X_test_3D, verbose=0)
                X_test_today  = X_test_3D[:, -1, :]
                X_test_hybrid = np.concatenate([X_test_latent, X_test_today], axis=1)

                xgb_pred_scaled = xgb_model.predict(X_test_hybrid)
                xgb_pred_ret     = dt.target_scaler.inverse_transform(xgb_pred_scaled)
                xgb_pred_accum.append(xgb_pred_ret)

            print(f"      ✅ Hoàn thành dự báo out-of-sample {year}!")

        # ── BƯỚC 4: Gộp kết quả và chạy Trading Simulation ──────────────────────
        trans_returns_all = np.concatenate(trans_pred_accum, axis=0)
        xgb_returns_all   = np.concatenate(xgb_pred_accum, axis=0) if not trans_only else None

        # Sanity Check
        assert len(trans_returns_all) == len(df_test_all), \
            f"Lệch độ dài: dự báo={len(trans_returns_all)}, thực tế={len(df_test_all)}"

        print(f"\n📊 [SIMULATION] Chạy giả lập trên dữ liệu liên tục 2023 → 2026 ({len(df_test_all)} phiên)...")
        dates, equity, bh_equity, trades = run_simulation(
            df_test_all, df_test_extended_all,
            xgb_returns_all, trans_returns_all,
            ticker, commission_pct, slippage_pct,
            threshold_buy, threshold_sell
        )

        metrics = compute_metrics(equity, bh_equity, dates)
        win_rate, total_trades, profit_factor = compute_trade_metrics(trades, commission_pct)

        print(f"\n🏆 KẾT QUẢ WALK-FORWARD ROLLING BACKTEST (2023–2026):")
        print(f"   📈 Chiến lược : {metrics['strat_return']:+.2f}%")
        print(f"   📦 Buy&Hold  : {metrics['bh_return']:+.2f}%")
        print(f"   {ticker} Rolling: Strategy {metrics['strat_return']:+.2f}% vs B&H {metrics['bh_return']:+.2f}%")
        print(f"   📊 Sharpe    : {metrics['sharpe']:.2f}")
        print(f"   📉 Max DD    : {metrics['mdd']:.2f}% (B&H: {metrics['bh_mdd']:.2f}%)")
        print(f"   ⚖️  Calmar    : {metrics['calmar']:.2f}")
        print(f"   🔔 Số lệnh   : {total_trades}")
        print(f"   🥇 Win Rate  : {win_rate:.2f}%")
        print(f"   💵 Profit F.  : {profit_factor:.2f}")

        wf = run_walk_forward_evaluation(dates, equity, bh_equity)
        print(f"\n🎯 CHI TIẾT HIỆU SUẤT THEO PHÂN ĐOẠN WINDOWS (Out-Of-Sample):")
        print("-" * 95)
        print(f"{'Window':<40} | {'Strategy':<12} | {'B&H':<12} | {'Sharpe':<8} | {'MDD'}")
        print("-" * 95)
        for w in wf:
            print(f"{w['window']:<40} | {w['strat_return']:+11.2f}% | "
                  f"{w['bh_return']:+11.2f}% | {w['sharpe']:<8.2f} | {w['mdd']:.2f}%")
        print("-" * 95)

        # Lưu biểu đồ đường cong tài sản
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label="Rolling Walk-Forward Strategy", color='darkgreen', linewidth=2)
        plt.plot(dates, bh_equity, label="Buy & Hold", color='grey', linestyle='--', alpha=0.8)
        plt.title(f"Rolling Walk-Forward Equity Curve — {ticker}", fontsize=13, fontweight='bold')
        plt.xlabel("Ngày")
        plt.ylabel("Tài sản (VNĐ)")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plot_path = os.path.join(figures_dir, f'walk_forward_equity_curve_{ticker}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 Đã lưu biểu đồ tại: {plot_path}")


if __name__ == "__main__":
    main()
