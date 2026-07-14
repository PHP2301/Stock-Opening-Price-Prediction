import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import datetime, random, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data
from src.features import DataTransformer
from src.ai_models import (
    PositionalEmbedding, TimeDecayAttention,
    MultiTaskModel, UncertaintyWeightsLayer, build_transformer,
    build_xgboost_optimized,
)
from src.backtest_engine import (
    get_return_output, compute_metrics, compute_trade_metrics,
    run_simulation, run_walk_forward_evaluation,
)

def compute_r2(y_true, y_pred):
    """Tính R2 score bỏ qua NaNs."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) < 2:
        return 0.0
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < 1e-9:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

# run_walk_forward_backtest.py — Expanding window walk-forward, Hybrid mỗi fold.
#
# QUAN TRỌNG (đã xác nhận với user): ở mỗi fold (mỗi năm kiểm thử), sau khi
# Transformer của fold đó train/warm-start xong, ta TRAIN MỘT XGBOOST MỚI
# TRÊN CHÍNH latent embedding của Transformer fold đó — KHÔNG dùng XGBoost
# cố định từ model production. Điều này đảm bảo latent space của Transformer
# và input của XGBoost luôn đồng bộ, tránh lệch phân phối giữa các fold.
#
# Logic simulation/metrics dùng chung từ src/backtest_engine.py.


def main():
    TICKERS       = ["META"]
    threshold_buy = 0.0010

    VOL_FILTER_THRESHOLD = {
        'VNM.VN': None,
        'GOOGL':  1.2,
        'META':   None,
    }

    args_cleaned = sys.argv
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
    hold_days = 5
    if len(args_cleaned) > 3:
        try:
            hold_days = int(args_cleaned[3])
        except ValueError:
            pass

    figures_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    config_dir  = os.path.join(ROOT_DIR, 'config')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # META lên sàn giữa năm 2012, Train 4 năm (2012-2015), test năm đầu là 2016
    test_years = list(range(2016, datetime.datetime.now().year + 1))

    for ticker in TICKERS:
        print(f"\n{'='*80}")
        print(f"🔄 BẮT ĐẦU ROLLING WALK-FORWARD BACKTEST (HYBRID): {ticker}")
        print(f"{'='*80}")

        commission_pct = 0.0020 if "VNM" in ticker.upper() else 0.0010
        slippage_pct   = 0.0010 if "VNM" in ticker.upper() else 0.0005

        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date=datetime.datetime.now().strftime("%Y-%m-%d"))
        df['date'] = pd.to_datetime(df['date'])

        test_start_date = pd.to_datetime(f"{test_years[0]}-01-01")
        if df['date'].max() < test_start_date:
            print(f"❌ Dữ liệu mã {ticker} quá ngắn, không có dữ liệu sau {test_start_date.strftime('%Y-%m-%d')}!")
            continue

        test_start_idx = df[df['date'] >= test_start_date].index[0]
        N_test_total = len(df) - test_start_idx

        df_test_all = df.iloc[test_start_idx:].reset_index(drop=True)
        df_test_extended_all = df.iloc[
            test_start_idx : test_start_idx + N_test_total + 1
        ].reset_index(drop=True)

        # Compute volatility ratio on full historical df to prevent NaN at start of test slice
        hist_returns_full = df['close'].pct_change().fillna(0.0)
        vol_20d_full = hist_returns_full.rolling(20).std().fillna(0.02)
        vol_250d_full = hist_returns_full.rolling(250).std().fillna(0.02)
        vol_ratio_full = (vol_20d_full / (vol_250d_full + 1e-9)).values
        vol_ratio_test_all = vol_ratio_full[test_start_idx : test_start_idx + N_test_total]

        # ── Đọc tham số tối ưu ─────────────────────────────────────────
        d_model, num_heads, key_dim, dropout_rate, learning_rate, batch_size = \
            128, 8, 16, 0.3, 1e-4, 64
        cfg_path = os.path.join(config_dir, f'best_transformer_params_{ticker}.json')
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8-sig') as f:
                    p = json.load(f)
                d_model       = p.get('d_model', d_model)
                num_heads     = p.get('num_heads', p.get('heads', num_heads))
                key_dim       = p.get('key_dim', key_dim)
                dropout_rate  = p.get('dropout_rate', dropout_rate)
                learning_rate = p.get('learning_rate', learning_rate)
                batch_size    = p.get('batch_size', batch_size)
                print(f"🥇 Đọc cấu hình tối ưu: d_model={d_model}, heads={num_heads}, "
                      f"key_dim={key_dim}, batch_size={batch_size}")
            except Exception as e:
                print(f"⚠️ Lỗi đọc file config: {e}")

        final_pred_accum = []   # Hybrid (XGBoost) final returns mỗi fold
        actual_target_accum = [] # Actual target returns mỗi fold để tính R2
        previous_weights = None

        for year in test_years:
            train_start = pd.to_datetime(f"{year-4}-01-01")
            year_start = pd.to_datetime(f"{year}-01-01")
            year_end   = pd.to_datetime(f"{year}-12-31")

            # Rolling Window: Lấy đúng 4 năm trước năm test để Train
            train_df = df[(df['date'] >= train_start) & (df['date'] < year_start)].copy()
            test_df  = df[(df['date'] >= year_start) & (df['date'] <= year_end)].copy()

            if test_df.empty:
                print(f"   [INFO] Năm {year}: Không có dữ liệu kiểm thử, bỏ qua.")
                continue

            print(f"\n▶️ [WINDOW {year}] — Huấn luyện: {year-4} → {year-1} "
                  f"({len(train_df)} phiên) | Kiểm thử: {year} ({len(test_df)} phiên)...")

            # Cô lập scaling hoàn toàn cho fold này
            dt = DataTransformer(time_steps=45, num_features=42)

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

            # Tách validation cho Transformer
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

            # ── BƯỚC 1: Train/warm-start Transformer ───────────────────────
            if previous_weights is not None:
                print(f"      [WARM-START] Khởi tạo trọng số từ Window trước...")
                transformer_model = build_transformer(
                    input_shape=(X_train_3D.shape[1], X_train_3D.shape[2]),
                    d_model=d_model, num_heads=num_heads, key_dim=key_dim,
                    dropout_rate=dropout_rate, learning_rate=2e-5,
                )
                try:
                    transformer_model.set_weights(previous_weights)
                    epochs_to_run, patience_run = 25, 8
                except Exception as e:
                    print(f"      ⚠️ Không load được trọng số: {e}, huấn luyện từ đầu.")
                    transformer_model = build_transformer(
                        input_shape=(X_train_3D.shape[1], X_train_3D.shape[2]),
                        d_model=d_model, num_heads=num_heads, key_dim=key_dim,
                        dropout_rate=dropout_rate, learning_rate=learning_rate,
                    )
                    epochs_to_run, patience_run = 100, 15
            else:
                transformer_model = build_transformer(
                    input_shape=(X_train_3D.shape[1], X_train_3D.shape[2]),
                    d_model=d_model, num_heads=num_heads, key_dim=key_dim,
                    dropout_rate=dropout_rate, learning_rate=learning_rate,
                )
                epochs_to_run, patience_run = 100, 15

            callbacks_window = [
                EarlyStopping(monitor='val_loss', patience=patience_run,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                  min_lr=1e-6, verbose=0),
            ]

            transformer_model.fit(
                X_tr,
                {"output_return": y_tr, "output_spread": y_tr_spread},
                validation_data=(X_va, {"output_return": y_va, "output_spread": y_va_spread}),
                epochs=epochs_to_run, batch_size=batch_size,
                callbacks=callbacks_window, verbose=0,
            )

            previous_weights = transformer_model.get_weights()

            # ── BƯỚC 2: Train XGBoost MỚI trên latent embedding của Transformer fold này ──
            # (theo xác nhận của user — KHÔNG dùng XGBoost cố định từ model production)
            print(f"      🌳 [XGBOOST] Huấn luyện XGBoost mới trên latent embedding fold {year}...")
            _ = transformer_model(X_train_3D[:1])
            feature_extractor = tf.keras.models.Model(
                inputs=transformer_model.inputs,
                outputs=transformer_model.get_layer("latent_embedding").output,
            )
            X_train_latent = feature_extractor.predict(X_train_3D, verbose=0)
            X_train_today  = X_train_3D[:, -1, :]
            X_train_hybrid = np.concatenate([X_train_latent, X_train_today], axis=1)

            xgb_model_fold = build_xgboost_optimized(X_train_hybrid, y_train_3D)

            # ── BƯỚC 3: Dự báo Hybrid out-of-sample cho năm hiện tại ──────
            test_extended_df = pd.concat([train_df.tail(45), test_df], ignore_index=True)
            X_test_raw = dt.transform_df(test_extended_df).values

            X_test_scaled = dt.feature_scaler.transform(X_test_raw)
            # y không cần thiết ở đây vì ta chỉ cần X cho inference fold
            y_test_dummy = np.zeros((len(X_test_scaled), len(dt.target_cols)))

            X_test_3D, _, _ = dt.create_sliding_windows(X_test_scaled, y_test_dummy, None)

            X_test_latent = feature_extractor.predict(X_test_3D, verbose=0)
            X_test_today  = X_test_3D[:, -1, :]
            X_test_hybrid = np.concatenate([X_test_latent, X_test_today], axis=1)

            xgb_pred_scaled = xgb_model_fold.predict(X_test_hybrid)
            final_pred_ret   = dt.target_scaler.inverse_transform(xgb_pred_scaled)
            final_pred_accum.append(final_pred_ret)

            # Lưu actual targets để tính R2
            y_test_raw = test_df[dt.target_cols].values
            actual_target_accum.append(y_test_raw)

            print(f"      ✅ Hoàn thành dự báo Hybrid out-of-sample {year}!")

        if not final_pred_accum:
            print(f"❌ Không có fold nào chạy được cho {ticker}, bỏ qua.")
            continue

        # ── BƯỚC 4: Gộp kết quả và chạy Trading Simulation ────────────────
        final_returns_all = np.concatenate(final_pred_accum, axis=0)
        actual_returns_all = np.concatenate(actual_target_accum, axis=0)

        assert len(final_returns_all) == len(df_test_all), \
            f"Lệch độ dài: dự báo={len(final_returns_all)}, thực tế={len(df_test_all)}"

        cur_threshold_buy = 0.0050 if "VNM" in ticker.upper() else 0.0050
        if len(args_cleaned) > 2:
            try:
                cur_threshold_buy = float(args_cleaned[2])
            except ValueError:
                pass
        cur_threshold_sell = -cur_threshold_buy

        # Chạy giả lập 1: KHÔNG LỌC (None)
        print(f"\n📊 [SIMULATION 1] Chạy giả lập Hybrid KHÔNG LỌC (threshold=None)...")
        dates_no_flt, equity_no_flt, bh_equity_no_flt, trades_no_flt = run_simulation(
            df_test_all, df_test_extended_all,
            final_returns_all, ticker,
            commission_pct, slippage_pct,
            cur_threshold_buy, cur_threshold_sell,
            vol_ratio=vol_ratio_test_all,
            vol_filter_threshold=None,
            hold_days=hold_days,
        )
        metrics_no_flt = compute_metrics(equity_no_flt, bh_equity_no_flt, dates_no_flt)
        win_rate_no_flt, total_trades_no_flt, profit_factor_no_flt = compute_trade_metrics(trades_no_flt, commission_pct)
        wf_no_flt = run_walk_forward_evaluation(dates_no_flt, equity_no_flt, bh_equity_no_flt)

        # Chạy giả lập 2: CÓ LỌC (Nếu threshold khác None)
        vol_filter_thr = VOL_FILTER_THRESHOLD.get(ticker)
        if vol_filter_thr is not None:
            print(f"\n📊 [SIMULATION 2] Chạy giả lập Hybrid CÓ LỌC (threshold={vol_filter_thr})...")
            dates_flt, equity_flt, bh_equity_flt, trades_flt = run_simulation(
                df_test_all, df_test_extended_all,
                final_returns_all, ticker,
                commission_pct, slippage_pct,
                cur_threshold_buy, cur_threshold_sell,
                vol_ratio=vol_ratio_test_all,
                vol_filter_threshold=vol_filter_thr,
                hold_days=hold_days,
            )
            metrics_flt = compute_metrics(equity_flt, bh_equity_flt, dates_flt)
            win_rate_flt, total_trades_flt, profit_factor_flt = compute_trade_metrics(trades_flt, commission_pct)
            wf_flt = run_walk_forward_evaluation(dates_flt, equity_flt, bh_equity_flt)
        else:
            dates_flt, equity_flt, bh_equity_flt = dates_no_flt, equity_no_flt, bh_equity_no_flt
            metrics_flt, win_rate_flt, total_trades_flt, profit_factor_flt = metrics_no_flt, win_rate_no_flt, total_trades_no_flt, profit_factor_no_flt
            wf_flt = wf_no_flt

        # In kết quả so sánh tổng thể
        r2_t1 = compute_r2(actual_returns_all[:, 0], final_returns_all[:, 0])
        print(f"\n🏆 KẾT QUẢ WALK-FORWARD ROLLING BACKTEST HYBRID (2023–2026):")
        print(f"{'Chỉ số':<25} | {'KHÔNG LỌC (None)':<20} | {f'CÓ LỌC ({vol_filter_thr})':<20}")
        print("-" * 72)
        print(f"{'Tỷ suất Chiến lược':<25} | {metrics_no_flt['strat_return']:+18.2f}% | {metrics_flt['strat_return']:+18.2f}%")
        print(f"{'Tỷ suất Buy & Hold':<25} | {metrics_no_flt['bh_return']:+18.2f}% | {metrics_flt['bh_return']:+18.2f}%")
        print(f"{'Hệ số Sharpe':<25} | {metrics_no_flt['sharpe']:19.2f} | {metrics_flt['sharpe']:19.2f}")
        print(f"{'Max Drawdown':<25} | {metrics_no_flt['mdd']:18.2f}% | {metrics_flt['mdd']:18.2f}%")
        print(f"{'Model R^2 Score (T+1)':<25} | {r2_t1:19.4f} | {r2_t1:19.4f}")
        print(f"{'Số lệnh':<25} | {total_trades_no_flt:19d} | {total_trades_flt:19d}")
        print(f"{'Tỷ lệ thắng':<25} | {win_rate_no_flt:18.2f}% | {win_rate_flt:18.2f}%")
        print(f"{'Profit Factor':<25} | {profit_factor_no_flt:19.2f} | {profit_factor_flt:19.2f}")

        # In kết quả breakdown 3 Windows so sánh
        print(f"\n🎯 CHI TIẾT HIỆU SUẤT THEO PHÂN ĐOẠN WINDOWS (Out-Of-Sample):")
        print("-" * 122)
        print(f"{'Window':<35} | {'No Flt Return':<14} | {'No Flt Sharpe':<13} | {'Flt Return':<12} | {'Flt Sharpe':<11} | {'Model R^2':<10} | {'B&H'}")
        print("-" * 122)
        n_dates = len(dates_no_flt)
        w_len = n_dates // 3
        for i, (w_no, w_f) in enumerate(zip(wf_no_flt, wf_flt)):
            s = i * w_len
            e = n_dates if i == 2 else (i + 1) * w_len
            y_true_slice = actual_returns_all[s:e, 0]
            y_pred_slice = final_returns_all[s:e, 0]
            r2_slice = compute_r2(y_true_slice, y_pred_slice)
            print(f"{w_no['window']:<35} | {w_no['strat_return']:+13.2f}% | {w_no['sharpe']:<13.2f} | "
                  f"{w_f['strat_return']:+11.2f}% | {w_f['sharpe']:<11.2f} | {r2_slice:<10.4f} | {w_no['bh_return']:+11.2f}%")
        print("-" * 122)

        # Vẽ biểu đồ so sánh cả 2 đường cong tài sản
        currency_label = "VNĐ" if "VNM" in ticker.upper() else "USD"
        plt.figure(figsize=(12, 6))
        plt.plot(dates_no_flt, equity_no_flt, label="Hybrid Strategy (No Filter)",
                 color='blue', linewidth=1.5, alpha=0.8)
        if vol_filter_thr is not None:
            plt.plot(dates_flt, equity_flt, label=f"Hybrid Strategy (Vol Filter {vol_filter_thr})",
                     color='darkgreen', linewidth=2)
        plt.plot(dates_no_flt, bh_equity_no_flt, label="Buy & Hold", color='grey',
                 linestyle='--', alpha=0.8)
        plt.title(f"Rolling Walk-Forward Equity Curve (Hybrid) — {ticker}",
                  fontsize=13, fontweight='bold')
        plt.xlabel("Ngày")
        plt.ylabel(f"Tài sản ({currency_label})")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plot_path = os.path.join(figures_dir, f'walk_forward_equity_curve_{ticker}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"💾 Đã lưu biểu đồ tại: {plot_path}")

        # Lưu hiệu suất backtest vào JSON
        perf_path = os.path.join(config_dir, f'performance_metrics_{ticker}.json')
        current_drawdown = (equity_flt[-1] - max(equity_flt)) / max(equity_flt) if len(equity_flt) > 0 else 0.0
        perf_data = {
            'overall_win_rate': win_rate_flt / 100.0,
            'total_trades': total_trades_flt,
            'profit_factor': profit_factor_flt,
            'strat_return': metrics_flt['strat_return'],
            'bh_return': metrics_flt['bh_return'],
            'sharpe': metrics_flt['sharpe'],
            'max_drawdown': metrics_flt['mdd'],
            'r2_score_t1': r2_t1,
            'current_drawdown': current_drawdown,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(perf_data, f, indent=4)
        print(f"💾 Đã lưu hiệu suất backtest vào: {perf_path}")


if __name__ == "__main__":
    main()