import os, sys, datetime, random, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
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
    TICKERS       = ["VNM.VN", "GOOGL", "META"]
    threshold_buy = 0.0010

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

    figures_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    config_dir  = os.path.join(ROOT_DIR, 'config')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    test_years = [2023, 2024, 2025, 2026]

    for ticker in TICKERS:
        print(f"\n{'='*80}")
        print(f"🔄 BẮT ĐẦU ROLLING WALK-FORWARD BACKTEST (HYBRID): {ticker}")
        print(f"{'='*80}")

        commission_pct = 0.0020 if "VNM" in ticker.upper() else 0.0010
        slippage_pct   = 0.0010 if "VNM" in ticker.upper() else 0.0005

        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date="2026-05-20")
        df['date'] = pd.to_datetime(df['date'])

        test_start_date = pd.to_datetime("2023-01-01")
        if df['date'].max() < test_start_date:
            print(f"❌ Dữ liệu mã {ticker} quá ngắn, không có dữ liệu sau 2023-01-01!")
            continue

        test_start_idx = df[df['date'] >= test_start_date].index[0]
        N_test_total = len(df) - test_start_idx

        df_test_all = df.iloc[test_start_idx:].reset_index(drop=True)
        df_test_extended_all = df.iloc[
            test_start_idx : test_start_idx + N_test_total + 1
        ].reset_index(drop=True)

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
        previous_weights = None

        for year in test_years:
            year_start = pd.to_datetime(f"{year}-01-01")
            year_end   = pd.to_datetime(f"{year}-12-31")

            train_df = df[df['date'] < year_start].copy()
            test_df  = df[(df['date'] >= year_start) & (df['date'] <= year_end)].copy()

            if test_df.empty:
                print(f"   [INFO] Năm {year}: Không có dữ liệu kiểm thử, bỏ qua.")
                continue

            print(f"\n▶️ [WINDOW {year}] — Huấn luyện: 2012 → {year-1} "
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

            print(f"      ✅ Hoàn thành dự báo Hybrid out-of-sample {year}!")

        if not final_pred_accum:
            print(f"❌ Không có fold nào chạy được cho {ticker}, bỏ qua.")
            continue

        # ── BƯỚC 4: Gộp kết quả và chạy Trading Simulation ────────────────
        final_returns_all = np.concatenate(final_pred_accum, axis=0)

        assert len(final_returns_all) == len(df_test_all), \
            f"Lệch độ dài: dự báo={len(final_returns_all)}, thực tế={len(df_test_all)}"

        cur_threshold_buy = 0.0050 if "VNM" in ticker.upper() else 0.0010
        if len(args_cleaned) > 2:
            try:
                cur_threshold_buy = float(args_cleaned[2])
            except ValueError:
                pass
        cur_threshold_sell = -cur_threshold_buy

        print(f"\n📊 [SIMULATION] Chạy giả lập Hybrid trên dữ liệu 2023 → 2026 "
              f"({len(df_test_all)} phiên)...")
        dates, equity, bh_equity, trades = run_simulation(
            df_test_all, df_test_extended_all,
            final_returns_all, ticker,
            commission_pct, slippage_pct,
            cur_threshold_buy, cur_threshold_sell,
        )

        metrics = compute_metrics(equity, bh_equity, dates)
        win_rate, total_trades, profit_factor = compute_trade_metrics(trades, commission_pct)

        print(f"\n🏆 KẾT QUẢ WALK-FORWARD ROLLING BACKTEST HYBRID (2023–2026):")
        print(f"   📈 Chiến lược : {metrics['strat_return']:+.2f}%")
        print(f"   📦 Buy&Hold  : {metrics['bh_return']:+.2f}%")
        print(f"   {ticker} Rolling Hybrid: Strategy {metrics['strat_return']:+.2f}% "
              f"vs B&H {metrics['bh_return']:+.2f}%")
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

        currency_label = "VNĐ" if "VNM" in ticker.upper() else "USD"
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label="Rolling Walk-Forward Hybrid Strategy",
                 color='darkgreen', linewidth=2)
        plt.plot(dates, bh_equity, label="Buy & Hold", color='grey',
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
        current_drawdown = (equity[-1] - max(equity)) / max(equity) if len(equity) > 0 else 0.0
        perf_data = {
            'overall_win_rate': win_rate / 100.0,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'strat_return': metrics['strat_return'],
            'bh_return': metrics['bh_return'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['mdd'],
            'current_drawdown': current_drawdown,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(perf_data, f, indent=4)
        print(f"💾 Đã lưu hiệu suất backtest vào: {perf_path}")


if __name__ == "__main__":
    main()