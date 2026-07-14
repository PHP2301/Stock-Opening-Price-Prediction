import os, sys, datetime, joblib, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data
from src.features import DataTransformer
from src.ai_models import (
    PositionalEmbedding, TimeDecayAttention,
    MultiTaskModel, UncertaintyWeightsLayer, CustomLambda,
)
from src.backtest_engine import (
    get_return_output, compute_metrics, compute_trade_metrics,
    run_simulation, run_walk_forward_evaluation,
)

# run_backtest.py — Static 80/20 split, Hybrid (Transformer + XGBoost), 42 features.
# Logic simulation/metrics dùng chung từ src/backtest_engine.py (không trùng lặp).
#
# Quy trình Hybrid:
#   1. Transformer dự báo trên X_test → lấy latent_embedding (40d)
#   2. Ghép latent_embedding + raw features hôm nay (42d) = 82d
#   3. XGBoost dự báo trên 82d → final_returns (kết quả CUỐI CÙNG)
#   4. run_simulation() nhận final_returns (đã qua Hybrid), không phải Transformer thô

ALERT_THRESHOLD = {
    'VNM.VN': 0.03,   # 3.0%
    'GOOGL':  0.025,  # 2.5%
    'META':   0.025,  # 2.5%
}

VOL_FILTER_THRESHOLD = {
    'VNM.VN': None,
    'GOOGL':  1.2,
    'META':   1.2,
}


def main():
    args_cleaned  = sys.argv
    TICKERS       = ["META"]
    cli_threshold = None

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

    custom_objects = {
        'PositionalEmbedding':     PositionalEmbedding,
        'TimeDecayAttention':      TimeDecayAttention,
        'UncertaintyWeightsLayer': UncertaintyWeightsLayer,
        'MultiTaskModel':          MultiTaskModel,
        'Lambda':                  CustomLambda,
    }

    for ticker in TICKERS:
        print(f"\n{'='*70}\n📊 BACKTEST (HYBRID): {ticker}\n{'='*70}")

        commission_pct = 0.0020 if "VNM" in ticker.upper() else 0.0010
        slippage_pct   = 0.0010 if "VNM" in ticker.upper() else 0.0005

        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date="2026-05-20")

        dt = DataTransformer(time_steps=45, num_features=42)
        X_scaled, y_scaled, y_spread_scaled = dt.fit_transform_train_only(df, train_ratio=0.8)
        X_3D, y_3D, y_spread_3D = dt.create_sliding_windows(X_scaled, y_scaled, y_spread_scaled)

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

        # ── Resolve đường dẫn model (với fallback timestamp) ─────────────
        trans_path = os.path.join(models_dir, f'transformer_model_{ticker}.keras')
        xgb_path   = os.path.join(models_dir, f'xgboost_model_{ticker}.pkl')
        feat_path  = os.path.join(models_dir, f'feature_scaler_{ticker}.pkl')
        targ_path  = os.path.join(models_dir, f'target_scaler_{ticker}.pkl')

        import glob
        if not os.path.exists(trans_path):
            ts_models = glob.glob(os.path.join(models_dir, f'transformer_model_{ticker}_*.keras'))
            if ts_models:
                ts_models.sort(key=os.path.getmtime, reverse=True)
                trans_path = ts_models[0]
                print(f"   ℹ️ Dùng model Transformer timestamp mới nhất: {os.path.basename(trans_path)}")
            else:
                print(f"❌ Không tìm thấy model Transformer cho {ticker}. Chạy run_training_transformer.py trước.")
                continue

        if not os.path.exists(xgb_path):
            ts_xgbs = glob.glob(os.path.join(models_dir, f'xgboost_model_{ticker}_*.pkl'))
            ts_xgbs = [f for f in ts_xgbs if "feature_scaler" not in f and "target_scaler" not in f]
            if ts_xgbs:
                ts_xgbs.sort(key=os.path.getmtime, reverse=True)
                xgb_path = ts_xgbs[0]
                print(f"   ℹ️ Dùng model XGBoost timestamp mới nhất: {os.path.basename(xgb_path)}")
            else:
                print(f"❌ Không tìm thấy model XGBoost cho {ticker}. Chạy run_training_transformer.py trước.")
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

        transformer_model = tf.keras.models.load_model(
            trans_path, custom_objects=custom_objects, safe_mode=False
        )
        xgb_model = joblib.load(xgb_path)
        dt.feature_scaler = joblib.load(feat_path)
        dt.target_scaler  = joblib.load(targ_path)

        # ── Hybrid inference: Transformer latent → XGBoost ────────────────
        print("   [PREDICT] Đang chạy dự báo Hybrid trên tập test...")
        _ = transformer_model(X_test[:1])  # dummy forward để build .input nếu cần

        feature_extractor = tf.keras.models.Model(
            inputs=transformer_model.inputs,
            outputs=transformer_model.get_layer("latent_embedding").output,
        )
        X_test_latent = feature_extractor.predict(X_test, verbose=0)
        X_test_today  = X_test[:, -1, :]
        X_test_hybrid = np.concatenate([X_test_latent, X_test_today], axis=1)

        xgb_pred_scaled = xgb_model.predict(X_test_hybrid)
        final_returns    = dt.target_scaler.inverse_transform(xgb_pred_scaled)

         # Threshold: CLI override hoặc per-ticker default
        if cli_threshold is not None:
            cur_threshold_buy = cli_threshold
        else:
            cur_threshold_buy = 0.0050 if "VNM" in ticker.upper() else 0.0010
        cur_threshold_sell = -cur_threshold_buy
        print(f"   🎯 Threshold buy={cur_threshold_buy*100:.2f}% | sell={cur_threshold_sell*100:.2f}%")

        # Compute volatility ratio on full historical df to prevent NaN at start of test slice
        hist_returns_full = df['close'].pct_change().fillna(0.0)
        vol_20d_full = hist_returns_full.rolling(20).std().fillna(0.02)
        vol_250d_full = hist_returns_full.rolling(250).std().fillna(0.02)
        vol_ratio_full = (vol_20d_full / (vol_250d_full + 1e-9)).values

        # Align vol_ratio with df_test
        start_idx = dt.time_steps + test_start_idx
        end_idx = start_idx + len(X_test)
        vol_ratio_test = vol_ratio_full[start_idx:end_idx]

        vol_filter_thr = VOL_FILTER_THRESHOLD.get(ticker)
        print(f"   🛡️ Volatility Filter: threshold={vol_filter_thr}")

        dates, equity, bh_equity, trades = run_simulation(
            df_test, df_test_extended,
            final_returns, ticker,
            commission_pct, slippage_pct,
            cur_threshold_buy, cur_threshold_sell,
            vol_ratio=vol_ratio_test,
            vol_filter_threshold=vol_filter_thr,
        )

        metrics                  = compute_metrics(equity, bh_equity, dates)
        win_rate, total_lnhs, pf = compute_trade_metrics(trades, commission_pct)

        # Debug cảnh báo biến động Telegram
        alert_thr = ALERT_THRESHOLD.get(ticker, 0.03)
        last_atr_pct = (
            (df_test['high'].iloc[-14:].values - df_test['low'].iloc[-14:].values)
            / df_test['close'].iloc[-14:].values
        ).mean() * 100 if len(df_test) >= 14 else 0.0
        print(f"   🔍 [Telegram check] ATR={last_atr_pct:.2f}% (ngưỡng {alert_thr*100:.1f}%) — "
              f"{'🚨 CẢNH BÁO' if last_atr_pct >= alert_thr*100 else '✅ Bình thường'}")

        print(f"\n🏆 KẾT QUẢ BACKTEST HYBRID (Out-of-Sample):")
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

        currency_label = "VNĐ" if "VNM" in ticker.upper() else "USD"
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity,    label="Hybrid Strategy", color='darkgreen', linewidth=2)
        plt.plot(dates, bh_equity, label="Buy & Hold",      color='grey', linestyle='--', alpha=0.8)
        plt.title(f"Equity Curve — {ticker}", fontsize=13, fontweight='bold')
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