import sys
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

# Configure UTF-8 for console output to avoid Windows charmap encoding issues
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def kalman_filter(series: pd.Series, R: float = 0.01, Q: float = 1e-5) -> pd.Series:
    """Bộ lọc Kalman FORWARD (causal) — chỉ dùng thông tin quá khứ."""
    if len(series) == 0:
        return series
    xhat = series.iloc[0]
    P = 1.0
    smoothed = []
    for val in series:
        P_minus = P + Q
        K = P_minus / (P_minus + R)
        xhat = xhat + K * (val - xhat)
        P = (1.0 - K) * P_minus
        smoothed.append(xhat)
    return pd.Series(smoothed, index=series.index)


class DataTransformer:
    def __init__(self, time_steps: int = 45):
        self.time_steps = time_steps
        self.feature_scaler = StandardScaler()
        self.target_scaler  = StandardScaler()
        self.spread_scaler  = StandardScaler()

        self.feature_cols = [
            # Nhánh 1 — Giá & Động lượng (12)
            'gap_open', 'open_return', 'buying_pressure', 'shadow_ratio',
            'intraday_range', 'return_1d', 'return_2d', 'return_3d',
            'mom_5d', 'mom_10d', 'mom_20d', 'dist_ma50',
            # Nhánh 2 — Khối lượng & Biến động (6)
            'volume_change', 'volume_sma_ratio', 'volume_zscore',
            'ad_line_ratio', 'obv_zscore', 'vol_ratio',
            # Nhánh 3 — Kỹ thuật, Vĩ mô & Lịch (16)
            'rsi_14', 'macd_ratio', 'bb_position', 'adx_14', 'stoch_k',
            'efficiency_ratio', 'vix_lag1', 'bond_yield_lag1',
            'usdvnd_change', 'vnindex_return_lag1',
            'sp500_above_ma200', 'nasdaq_12m_return',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'is_quarter_end', 'days_before_tet',
            # Nhánh 4 — Dòng tiền & Cổ tức (8 mới)
            'mfi_14', 'dividend_flag', 'days_to_dividend', 'days_after_dividend',
            'foreign_net_buy_proxy', 'foreign_net_buy_5d', 'foreign_net_buy_20d',
            'self_net_buy_proxy'
        ]
        self.target_cols = ['target_return_1d', 'target_return_2d', 'target_return_3d']
        self.spread_cols = ['target_spread_1d', 'target_spread_2d', 'target_spread_3d']

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy().replace([np.inf, -np.inf], np.nan)

        if 'close_smoothed' not in df_copy.columns:
            df_copy['close_smoothed'] = kalman_filter(df_copy['close'])

        # ── Nhánh 1: Giá & Động lượng ────────────────────────────────
        df_copy['gap_open']    = df_copy['open'] / df_copy['close'].shift(1) - 1
        df_copy['open_return'] = df_copy['open'] / df_copy['open'].shift(1) - 1
        df_copy['buying_pressure'] = (
            (df_copy['close_smoothed'] - df_copy['low'])
            / (df_copy['high'] - df_copy['low'] + 1e-9)
        )
        df_copy['shadow_ratio'] = (
            (df_copy['high'] - df_copy['close_smoothed'])
            / (df_copy['close_smoothed'] - df_copy['low'] + 1e-9)
        )
        df_copy['intraday_range'] = (
            (df_copy['high'] - df_copy['low']) / df_copy['close_smoothed']
        )
        df_copy['return_1d'] = (
            df_copy['close_smoothed'].shift(1) / df_copy['close_smoothed'].shift(2) - 1
        )
        df_copy['return_2d'] = (
            df_copy['close_smoothed'].shift(2) / df_copy['close_smoothed'].shift(3) - 1
        )
        df_copy['return_3d'] = (
            df_copy['close_smoothed'].shift(3) / df_copy['close_smoothed'].shift(4) - 1
        )
        df_copy['mom_5d']  = df_copy['close_smoothed'] / df_copy['close_smoothed'].shift(5)  - 1
        df_copy['mom_10d'] = df_copy['close_smoothed'] / df_copy['close_smoothed'].shift(10) - 1
        df_copy['mom_20d'] = df_copy['close_smoothed'] / df_copy['close_smoothed'].shift(20) - 1
        df_copy['dist_ma50'] = (
            df_copy['close_smoothed']
            / df_copy['close_smoothed'].rolling(50).mean() - 1
        )

        # ── Nhánh 2: Khối lượng & Biến động ──────────────────────────
        df_copy['volume_change']    = df_copy['volume'].pct_change()
        df_copy['volume_sma_ratio'] = (
            df_copy['volume'] / (df_copy['volume'].rolling(20).mean() + 1e-9)
        )
        mean_vol = df_copy['volume'].rolling(20).mean()
        std_vol  = df_copy['volume'].rolling(20).std()
        df_copy['volume_zscore'] = (df_copy['volume'] - mean_vol) / (std_vol + 1e-9)
        df_copy['ad_line_ratio'] = (
            (df_copy['close_smoothed'] - df_copy['low'])
            - (df_copy['high'] - df_copy['close_smoothed'])
        ) / (df_copy['high'] - df_copy['low'] + 1e-9)

        obv_dir = np.where(
            df_copy['close_smoothed'].diff() > 0, 1,
            np.where(df_copy['close_smoothed'].diff() < 0, -1, 0)
        )
        obv       = (obv_dir * df_copy['volume']).cumsum()
        delta_obv = obv.diff(5)
        df_copy['obv_zscore'] = delta_obv / (delta_obv.rolling(20).std() + 1e-9)

        pct_change = df_copy['close_smoothed'].pct_change()
        df_copy['vol_ratio'] = (
            pct_change.rolling(5).std() / (pct_change.rolling(60).std() + 1e-9)
        )

        # ── Nhánh 3: Kỹ thuật, Vĩ mô & Lịch ─────────────────────────
        df_copy['rsi_14'] = ta.rsi(df_copy['close_smoothed'], length=14)

        macd_df = ta.macd(df_copy['close_smoothed'], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            df_copy['macd_ratio'] = macd_df.iloc[:, 0] / (macd_df.iloc[:, 2] + 1e-9)
        else:
            df_copy['macd_ratio'] = 0.0

        bb_df = ta.bbands(df_copy['close_smoothed'], length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            df_copy['bb_position'] = (
                (df_copy['close_smoothed'] - bb_df.iloc[:, 0])
                / (bb_df.iloc[:, 2] - bb_df.iloc[:, 0] + 1e-9)
            )
        else:
            df_copy['bb_position'] = 0.5

        adx_df = ta.adx(df_copy['high'], df_copy['low'], df_copy['close_smoothed'], length=14)
        df_copy['adx_14'] = adx_df.iloc[:, 0] if adx_df is not None else 20.0

        stoch_df = ta.stoch(df_copy['high'], df_copy['low'], df_copy['close_smoothed'], fast_k=14)
        df_copy['stoch_k'] = stoch_df.iloc[:, 0] if stoch_df is not None else 50.0

        daily_changes = df_copy['close_smoothed'].diff()
        df_copy['efficiency_ratio'] = (
            (df_copy['close_smoothed'] - df_copy['close_smoothed'].shift(10)).abs()
            / (daily_changes.abs().rolling(10).sum() + 1e-9)
        )

        # ── Macro columns từ data_loader (passthrough) ────────────────
        # FIX: các cột macro đã tính trong data_loader, chỉ cần copy qua
        for col in ['vix_lag1', 'bond_yield_lag1', 'usdvnd_change',
                    'vnindex_return_lag1', 'sp500_above_ma200', 'nasdaq_12m_return',
                    'day_of_week_sin', 'day_of_week_cos',
                    'month_sin', 'month_cos', 'is_quarter_end', 'days_before_tet']:
            if col in df.columns:
                df_copy[col] = df[col].reset_index(drop=True).values
            else:
                df_copy[col] = 0.0

        # Passthrough các cột cổ tức từ data_loader (mặc định nếu thiếu)
        for col in ['dividend_flag', 'days_to_dividend', 'days_after_dividend']:
            if col in df.columns:
                df_copy[col] = df[col].reset_index(drop=True).values
            else:
                df_copy[col] = 60.0 if col != 'dividend_flag' else 0.0

        # ── Nhánh 4: Dòng tiền & Cổ tức (Tính toán bổ sung) ───────────
        # MFI 14 ngày
        df_copy['mfi_14'] = ta.mfi(df_copy['high'], df_copy['low'], df_copy['close_smoothed'], df_copy['volume'], length=14)
        df_copy['mfi_14'] = df_copy['mfi_14'].fillna(50.0)

        # Foreign Net Buy Proxy
        clv = (2 * df_copy['close_smoothed'] - df_copy['high'] - df_copy['low']) / (df_copy['high'] - df_copy['low'] + 1e-9)
        df_copy['foreign_net_buy_proxy'] = df_copy['volume_zscore'] * clv
        df_copy['foreign_net_buy_5d'] = df_copy['foreign_net_buy_proxy'].rolling(5).mean()
        df_copy['foreign_net_buy_20d'] = df_copy['foreign_net_buy_proxy'].rolling(20).mean()

        # Self Net Buy Proxy (MFI Divergence + Volume Spike)
        mfi_change = df_copy['mfi_14'].diff()
        price_change = df_copy['close_smoothed'].pct_change()
        sign_price_opposite = -np.sign(price_change)
        mfi_divergence = mfi_change * sign_price_opposite
        large_vol = np.where(df_copy['volume_zscore'] > 2.0, 1.0, 0.0)
        df_copy['self_net_buy_proxy'] = mfi_divergence * (1.0 + large_vol)

        # ── Trích xuất & điền khuyết ──────────────────────────────────
        df_out = pd.DataFrame(index=df.index)
        for col in self.feature_cols:
            df_out[col] = df_copy[col] if col in df_copy.columns else 0.0

        df_out = df_out.replace([np.inf, -np.inf], np.nan)
        for col in self.feature_cols:
            df_out[col] = df_out[col].ffill().bfill().fillna(0.0)

        df_out['days_to_dividend']    = df_out['days_to_dividend'].fillna(60.0)
        df_out['days_after_dividend'] = df_out['days_after_dividend'].fillna(60.0)

        return df_out[self.feature_cols]

    def fit_transform_train_only(
        self, df: pd.DataFrame, train_ratio: float = 0.8, purge_gap: int = 45
    ):
        df_feats = self.transform_df(df)
        X_raw = df_feats.values
        y_raw = df[self.target_cols].values

        total_windows    = len(df_feats) - self.time_steps
        split_idx_window = int(total_windows * train_ratio)
        split_idx_raw    = split_idx_window + self.time_steps

        self.feature_scaler.fit(X_raw[:split_idx_raw])
        self.target_scaler.fit(y_raw[:split_idx_raw])

        X_scaled = self.feature_scaler.transform(X_raw)
        y_scaled = self.target_scaler.transform(y_raw)

        y_spread_scaled = None
        if all(col in df.columns for col in self.spread_cols):
            y_spread_raw = df[self.spread_cols].values
            self.spread_scaler.fit(y_spread_raw[:split_idx_raw])
            y_spread_scaled = self.spread_scaler.transform(y_spread_raw)

        return X_scaled, y_scaled, y_spread_scaled

    def fit_transform_data(self, df: pd.DataFrame):
        """Fit+transform toàn bộ — chỉ dùng cho testing nhanh."""
        df_feats = self.transform_df(df)
        X_raw = df_feats.values
        y_raw = df[self.target_cols].values

        X_scaled = self.feature_scaler.fit_transform(X_raw)
        y_scaled = self.target_scaler.fit_transform(y_raw)

        y_spread_scaled = None
        if all(col in df.columns for col in self.spread_cols):
            y_spread_raw = df[self.spread_cols].values
            y_spread_scaled = self.spread_scaler.fit_transform(y_spread_raw)

        return X_scaled, y_scaled, y_spread_scaled

    def create_sliding_windows(self, X_scaled, y_scaled, y_spread_scaled=None):
        X_3D, y_3D, y_spread_3D = [], [], []
        for i in range(self.time_steps, len(X_scaled)):
            X_3D.append(X_scaled[i - self.time_steps: i])
            y_3D.append(y_scaled[i])
            if y_spread_scaled is not None:
                y_spread_3D.append(y_spread_scaled[i])
        if y_spread_scaled is not None:
            return np.array(X_3D), np.array(y_3D), np.array(y_spread_3D)
        return np.array(X_3D), np.array(y_3D), None

    def split_train_test_chronological(
        self, df, X_3D, y_3D, y_spread_3D=None,
        train_ratio=0.8, purge_gap=45
    ):
        total     = len(X_3D)
        split_idx = int(total * train_ratio)

        X_train = X_3D[:split_idx]
        y_train = y_3D[:split_idx]

        test_start = min(split_idx + purge_gap, total)
        X_test = X_3D[test_start:]
        y_test = y_3D[test_start:]

        y_train_spread = y_spread_3D[:split_idx]  if y_spread_3D is not None else None
        y_test_spread  = y_spread_3D[test_start:] if y_spread_3D is not None else None

        df_align   = df.iloc[self.time_steps:].reset_index(drop=True)
        # FIX: iloc[test_start:] bao gồm đúng rows tương ứng với X_test
        y_test_raw = df_align.iloc[test_start:][self.target_cols].values

        # Loại bỏ các dòng có target là NaN khỏi tập test (do ngày giao dịch cuối cùng chưa có giá tương lai)
        non_nan_mask = ~np.isnan(y_test_raw).any(axis=1)
        X_test = X_test[non_nan_mask]
        y_test = y_test[non_nan_mask]
        y_test_raw = y_test_raw[non_nan_mask]
        if y_test_spread is not None:
            y_test_spread = y_test_spread[non_nan_mask]

        print(f"📊 Split {int(round(train_ratio*100))}/{int(round((1-train_ratio)*100))} "
              f"(Purge Gap: {purge_gap}):")
        print(f"   🔹 Train: {X_train.shape[0]} mẫu")
        print(f"   🔸 Test : {X_test.shape[0]} mẫu")

        return (X_train, y_train, X_test, y_test,
                y_test_raw, y_train_spread, y_test_spread)

    def split_train_test_by_year(self, df, X_3D, y_3D, y_spread_3D=None):
        df_align = df.iloc[self.time_steps:].reset_index(drop=True)
        df_align['date'] = pd.to_datetime(df_align['date'])
        train_mask = df_align['date'].dt.year <= 2023
        test_mask  = df_align['date'].dt.year >= 2024
        X_train = X_3D[train_mask]; y_train = y_3D[train_mask]
        X_test  = X_3D[test_mask];  y_test  = y_3D[test_mask]
        y_train_spread = y_spread_3D[train_mask] if y_spread_3D is not None else None
        y_test_spread  = y_spread_3D[test_mask]  if y_spread_3D is not None else None
        y_test_raw = df_align.loc[test_mask, self.target_cols].values

        # Loại bỏ các dòng có target là NaN khỏi tập test
        non_nan_mask = ~np.isnan(y_test_raw).any(axis=1)
        X_test = X_test[non_nan_mask]
        y_test = y_test[non_nan_mask]
        y_test_raw = y_test_raw[non_nan_mask]
        if y_test_spread is not None:
            y_test_spread = y_test_spread[non_nan_mask]

        return (X_train, y_train, X_test, y_test,
                y_test_raw, y_train_spread, y_test_spread)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("=== CHẠY BIẾN ĐỔI ĐẶC TRƯNG ===")
    from src.data_loader import fetch_and_prepare_data
    for ticker in ["VNM.VN", "GOOGL", "META"]:
        print(f"\n🔬 {ticker}")
        try:
            df = fetch_and_prepare_data(ticker, "2015-01-01", "2026-05-20")
            t  = DataTransformer(time_steps=45)
            X, y, _ = t.fit_transform_train_only(df)
            X3, y3, _ = t.create_sliding_windows(X, y)
            print(f"   => {X3.shape}")
        except Exception as e:
            print(f"   => Lỗi: {e}")