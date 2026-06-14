import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import DataTransformer, kalman_filter

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Tạo dữ liệu giả lập
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range(start="2025-01-01", periods=n_samples, freq="D")
        
        self.df = pd.DataFrame({
            "date": dates,
            "open": np.random.uniform(50000, 60000, n_samples),
            "high": np.random.uniform(60000, 65000, n_samples),
            "low": np.random.uniform(45000, 50000, n_samples),
            "close": np.random.uniform(50000, 60000, n_samples),
            "close_smoothed": np.random.uniform(50000, 60000, n_samples),
            "volume": np.random.uniform(100000, 1000000, n_samples),
            "vix": np.random.uniform(10, 30, n_samples),
            "bond_yield_10y": np.random.uniform(2.0, 5.0, n_samples),
            "dollar_index_change": np.random.uniform(-0.02, 0.02, n_samples),
            "usdvnd": np.random.uniform(25000, 26000, n_samples),
            "vnindex": np.random.uniform(1100000, 1200000, n_samples),
            "volume_zscore": np.random.uniform(-2, 2, n_samples)
        })
        
        # Thêm các cột mục tiêu
        for h in [1, 2, 3]:
            self.df[f"target_return_{h}d"] = self.df["open"].shift(-h) / self.df["close"] - 1.0
            self.df[f"target_spread_{h}d"] = (self.df["high"].shift(-h) - self.df["low"].shift(-h)) / self.df["close"]
        self.df["days_before_tet"] = 30
        self.df["is_quarter_end"] = 0
        self.df["day_of_week_sin"] = 0.5
        self.df["day_of_week_cos"] = 0.5
        self.df["month_sin"] = 0.5
        self.df["month_cos"] = 0.5
        self.df["dividend_flag"] = 0.0
        self.df["days_to_dividend"] = 60.0
        self.df["days_after_dividend"] = 60.0
        
        # Điền các đặc trưng thô khác cần thiết cho transform_df nếu có
        # Ở đây ta sẽ giả định dữ liệu đã đi qua data_loader và chứa các đặc trưng thô
        self.transformer = DataTransformer(time_steps=10)

    def test_kalman_filter(self):
        """Kiểm tra bộ lọc Kalman hoạt động bình thường trên Series"""
        s = pd.Series([1.0, 1.2, 1.1, 1.3, 1.4])
        smoothed = kalman_filter(s)
        self.assertEqual(len(smoothed), len(s))
        self.assertIsInstance(smoothed, pd.Series)

    def test_sliding_windows(self):
        """Kiểm tra việc chuyển đổi cấu trúc 3D sliding window"""
        X_dummy = np.random.randn(50, 42)
        y_dummy = np.random.randn(50, 3)
        ys_dummy = np.random.randn(50, 3)
        
        X_3d, y_3d, ys_3d = self.transformer.create_sliding_windows(X_dummy, y_dummy, ys_dummy)
        
        expected_n = 50 - self.transformer.time_steps
        self.assertEqual(X_3d.shape, (expected_n, self.transformer.time_steps, 42))
        self.assertEqual(y_3d.shape, (expected_n, 3))
        self.assertEqual(ys_3d.shape, (expected_n, 3))

    def test_fit_transform_train_only(self):
        """Kiểm tra việc chuẩn hóa chỉ dùng phân phối của tập Train (không Leak)"""
        # Điền khuyết NaN/inf để sẵn sàng transform_df
        df_clean = self.df.dropna().copy()
        X_scaled, y_scaled, y_spread_scaled = self.transformer.fit_transform_train_only(df_clean, train_ratio=0.8)
        
        # Verify shapes are correct
        n_samples = len(df_clean)
        self.assertEqual(X_scaled.shape, (n_samples, len(self.transformer.feature_cols)))
        self.assertEqual(y_scaled.shape, (n_samples, 3))
        self.assertEqual(y_spread_scaled.shape, (n_samples, 3))

if __name__ == '__main__':
    unittest.main()
