import os, sys, joblib, json, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.experimental.enable_op_determinism()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader import fetch_and_prepare_data
from src.features import DataTransformer
from src.ai_models import (
    PositionalEmbedding, TimeDecayAttention,
    MultiTaskModel, UncertaintyWeightsLayer, CustomLambda,
)
from src.backtest_engine import get_return_output

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def run_evaluation():
    TICKERS = ["META"]
    models_dir  = os.path.join(ROOT_DIR, 'models')
    figures_dir = os.path.join(ROOT_DIR, 'reports', 'figures')
    config_dir  = os.path.join(ROOT_DIR, 'config')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    custom_objects = {
        'PositionalEmbedding':     PositionalEmbedding,
        'TimeDecayAttention':      TimeDecayAttention,
        'UncertaintyWeightsLayer': UncertaintyWeightsLayer,
        'MultiTaskModel':          MultiTaskModel,
        'Lambda':                  CustomLambda,
    }

    results_summary = {}

    for ticker in TICKERS:
        print(f"\n==========================================")
        print(f"📊 ĐÁNH GIÁ THỐNG KÊ (WILCOXON & BOOTSTRAP): {ticker}")
        print(f"==========================================")

        df = fetch_and_prepare_data(ticker, start_date="2012-01-01", end_date="2026-05-20")

        dt = DataTransformer(time_steps=45, num_features=42)
        X_scaled, y_scaled, y_spread_scaled = dt.fit_transform_train_only(df, train_ratio=0.8)
        X_3D, y_3D, y_spread_3D = dt.create_sliding_windows(X_scaled, y_scaled, y_spread_scaled)

        _, _, X_test, _, y_test_raw, _, _ = dt.split_train_test_chronological(
            df, X_3D, y_3D, y_spread_3D, train_ratio=0.8
        )

        # 1. Load Models & Scalers
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
            else:
                print(f"❌ Không tìm thấy model Transformer cho {ticker}.")
                continue

        if not os.path.exists(xgb_path):
            ts_xgbs = glob.glob(os.path.join(models_dir, f'xgboost_model_{ticker}_*.pkl'))
            ts_xgbs = [f for f in ts_xgbs if "feature_scaler" not in f and "target_scaler" not in f]
            if ts_xgbs:
                ts_xgbs.sort(key=os.path.getmtime, reverse=True)
                xgb_path = ts_xgbs[0]
            else:
                print(f"❌ Không tìm thấy model XGBoost cho {ticker}.")
                continue

        if not os.path.exists(feat_path) or not os.path.exists(targ_path):
            print(f"❌ Không tìm thấy feature/target scaler cho {ticker}.")
            continue

        transformer_model = tf.keras.models.load_model(
            trans_path, custom_objects=custom_objects, safe_mode=False
        )
        xgb_model = joblib.load(xgb_path)
        dt.feature_scaler = joblib.load(feat_path)
        dt.target_scaler  = joblib.load(targ_path)

        # 2. Model Inference
        # Dummy forward to build model if needed
        _ = transformer_model(X_test[:1])

        # Get Transformer raw predictions (baseline)
        trans_preds = transformer_model.predict(X_test, verbose=0)
        trans_pred_scaled = get_return_output(trans_preds)
        trans_pred_raw = dt.target_scaler.inverse_transform(trans_pred_scaled)

        # Get Hybrid predictions (Transformer latent + XGBoost stacking)
        feature_extractor = tf.keras.models.Model(
            inputs=transformer_model.inputs,
            outputs=transformer_model.get_layer("latent_embedding").output,
        )
        X_test_latent = feature_extractor.predict(X_test, verbose=0)
        X_test_today  = X_test[:, -1, :]
        X_test_hybrid = np.concatenate([X_test_latent, X_test_today], axis=1)

        xgb_pred_scaled = xgb_model.predict(X_test_hybrid)
        hybrid_pred_raw = dt.target_scaler.inverse_transform(xgb_pred_scaled)

        # Target Return T+1 is at index 0
        y_true_1d = y_test_raw[:, 0]
        pred_trans_1d = trans_pred_raw[:, 0]
        pred_hybrid_1d = hybrid_pred_raw[:, 0]

        # Calculate errors
        ae_trans = np.abs(pred_trans_1d - y_true_1d)
        ae_hybrid = np.abs(pred_hybrid_1d - y_true_1d)

        mae_trans_val = np.mean(ae_trans)
        mae_hybrid_val = np.mean(ae_hybrid)

        print(f"   📉 MAE Pure Transformer: {mae_trans_val*100:.4f}%")
        print(f"   📉 MAE Hybrid (Proposed): {mae_hybrid_val*100:.4f}%")

        # 3. Wilcoxon Signed-Rank Test
        # Test if the absolute errors of the Hybrid model are significantly smaller than the Transformer model
        wilcox_stat, wilcox_pval = wilcoxon(ae_hybrid, ae_trans, alternative='less')
        print(f"   🧪 Wilcoxon Signed-Rank Test p-value: {wilcox_pval:.6f}")
        is_significant = wilcox_pval < 0.05
        print(f"   💡 Có ý nghĩa thống kê (p < 0.05)? {'CÓ (Đạt yêu cầu)' if is_significant else 'KHÔNG'}")

        # 4. Bootstrap Confidence Interval (95%)
        # Resample with replacement to estimate the distribution of MAE
        n_bootstraps = 1000
        boot_mae_trans = []
        boot_mae_hybrid = []
        n_samples = len(y_true_1d)

        for _ in range(n_bootstraps):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_mae_trans.append(np.mean(ae_trans[indices]))
            boot_mae_hybrid.append(np.mean(ae_hybrid[indices]))

        boot_mae_trans = np.array(boot_mae_trans)
        boot_mae_hybrid = np.array(boot_mae_hybrid)

        ci_trans_low, ci_trans_high = np.percentile(boot_mae_trans, [2.5, 97.5])
        ci_hybrid_low, ci_hybrid_high = np.percentile(boot_mae_hybrid, [2.5, 97.5])

        print(f"   📊 95% CI MAE Transformer: [{ci_trans_low*100:.4f}%, {ci_trans_high*100:.4f}%]")
        print(f"   📊 95% CI MAE Hybrid:      [{ci_hybrid_low*100:.4f}%, {ci_hybrid_high*100:.4f}%]")

        results_summary[ticker] = {
            'mae_trans': float(mae_trans_val),
            'mae_hybrid': float(mae_hybrid_val),
            'wilcox_pval': float(wilcox_pval),
            'is_significant': bool(is_significant),
            'ci_trans': [float(ci_trans_low), float(ci_trans_high)],
            'ci_hybrid': [float(ci_hybrid_low), float(ci_hybrid_high)]
        }

        # 5. Plotting Wilcoxon absolute errors distribution
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=[ae_trans * 100, ae_hybrid * 100], palette=["#FFA07A", "#20B2AA"])
        plt.xticks([0, 1], ['Pure Transformer', 'Hybrid (Proposed)'])
        plt.ylabel('Absolute Error (%)')
        plt.title(f'Wilcoxon Signed-Rank Test: Absolute Error Comparison ({ticker})\np-value: {wilcox_pval:.6e}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        wilcox_plot_path = os.path.join(figures_dir, f'wilcoxon_test_errors_{ticker}.png')
        plt.savefig(wilcox_plot_path, dpi=300)
        plt.close()
        print(f"   💾 Đã lưu biểu đồ Wilcoxon tại: {wilcox_plot_path}")

        # 6. Plotting Bootstrap distributions of MAE
        plt.figure(figsize=(10, 5))
        sns.kdeplot(boot_mae_trans * 100, label='Pure Transformer', fill=True, color='red', alpha=0.3)
        sns.kdeplot(boot_mae_hybrid * 100, label='Hybrid (Proposed)', fill=True, color='green', alpha=0.3)
        
        # Vertical lines for Mean
        plt.axvline(mae_trans_val * 100, color='darkred', linestyle='--', linewidth=1.5, label=f'Mean Trans ({mae_trans_val*100:.3f}%)')
        plt.axvline(mae_hybrid_val * 100, color='darkgreen', linestyle='--', linewidth=1.5, label=f'Mean Hybrid ({mae_hybrid_val*100:.3f}%)')

        # Add CI bands
        plt.axvspan(ci_trans_low * 100, ci_trans_high * 100, color='red', alpha=0.08, label='95% CI Transformer')
        plt.axvspan(ci_hybrid_low * 100, ci_hybrid_high * 100, color='green', alpha=0.08, label='95% CI Hybrid')

        plt.xlabel('MAE (%)')
        plt.ylabel('Density')
        plt.title(f'Bootstrap MAE Distribution and 95% Confidence Intervals ({ticker})', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        boot_plot_path = os.path.join(figures_dir, f'bootstrap_confidence_interval_{ticker}.png')
        plt.savefig(boot_plot_path, dpi=300)
        plt.close()
        print(f"   💾 Đã lưu biểu đồ Bootstrap tại: {boot_plot_path}")

    # Write summary JSON
    summary_path = os.path.join(config_dir, 'statistical_eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Đã hoàn thành toàn bộ đánh giá thống kê và lưu kết quả tại: {summary_path}")

if __name__ == "__main__":
    run_evaluation()
