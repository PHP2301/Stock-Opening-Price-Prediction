import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def kalman_filter(series: pd.Series, R: float = 0.01, Q: float = 1e-5) -> pd.Series:
    """
    Bộ lọc Kalman làm mịn dữ liệu chuỗi thời gian bằng thuật toán cập nhật trạng thái đệ quy.
    """
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
        """
        Khởi tạo bộ biến đổi dữ liệu.
        :param time_steps: Số lượng phiên giao dịch quá khứ AI dùng để nhìn lại (mặc định 45 phiên).
        """
        self.time_steps = time_steps
        # Khởi tạo 2 bộ scaler độc lập: 1 cho các đặc trưng đầu vào, 1 riêng cho biến mục tiêu (Target)
        # Sử dụng StandardScaler để tránh bị ảnh hưởng bởi các giá trị ngoại lai (outliers)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.spread_scaler = StandardScaler()
        
        # Định nghĩa 34 đặc trưng cơ bản
        self.feature_cols = [
            # Nhánh 1 — Giá & Động lượng (12)
            'gap_open', 'open_return', 'buying_pressure', 'shadow_ratio', 'intraday_range',
            'return_1d', 'return_2d', 'return_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'dist_ma50',
            # Nhánh 2 — Khối lượng & Biến động (6)
            'volume_change', 'volume_sma_ratio', 'volume_zscore', 'ad_line_ratio', 'obv_zscore', 'vol_ratio',
            # Nhánh 3 — Kỹ thuật, Vĩ mô & Lịch (16)
            'rsi_14', 'macd_ratio', 'bb_position', 'adx_14', 'stoch_k', 'efficiency_ratio',
            'vix_lag1', 'bond_yield_lag1', 'usdvnd_change', 'vnindex_return_lag1',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'is_quarter_end', 'days_before_tet'
        ]
        self.target_col = 'target_return'
        self.spread_col = 'target_spread'

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán các đặc trưng tỷ lệ (Stationary Ratios) từ dữ liệu thô.
        Đảm bảo tính đồng nhất giữa huấn luyện và dự đoán trực tuyến.
        """
        df_copy = df.copy()
        # Chuyển đổi vô cực (inf) thành NaN trước khi tính toán
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
        
        # Nhánh 1: Giá & Động lượng
        df_copy['gap_open'] = df_copy['open'] / df_copy['close'].shift(1) - 1
        df_copy['open_return'] = df_copy['open'] / df_copy['open'].shift(1) - 1
        df_copy['buying_pressure'] = (df_copy['close_smoothed'] - df_copy['low']) / (df_copy['high'] - df_copy['low'] + 1e-9)
        df_copy['shadow_ratio'] = (df_copy['high'] - df_copy['close_smoothed']) / (df_copy['close_smoothed'] - df_copy['low'] + 1e-9)
        df_copy['intraday_range'] = (df_copy['high'] - df_copy['low']) / df_copy['close_smoothed']
        df_copy['return_1d'] = df_copy['close_smoothed'].shift(1) / df_copy['close_smoothed'].shift(2) - 1
        df_copy['return_2d'] = df_copy['close_smoothed'].shift(2) / df_copy['close_smoothed'].shift(3) - 1
        df_copy['return_3d'] = df_copy['close_smoothed'].shift(3) / df_copy['close_smoothed'].shift(4) - 1
        df_copy['mom_5d'] = df_copy['close_smoothed'] / df_copy['close_smoothed'].shift(5) - 1
        df_copy['mom_10d'] = df_copy['close_smoothed'] / df_copy['close_smoothed'].shift(10) - 1
        df_copy['mom_20d'] = df_copy['close_smoothed'] / df_copy['close_smoothed'].shift(20) - 1
        df_copy['dist_ma50'] = df_copy['close_smoothed'] / df_copy['close_smoothed'].rolling(50).mean() - 1

        # Nhánh 2: Khối lượng & Biến động
        df_copy['volume_change'] = df_copy['volume'].pct_change()
        df_copy['volume_sma_ratio'] = df_copy['volume'] / (df_copy['volume'].rolling(20).mean() + 1e-9)
        mean_vol = df_copy['volume'].rolling(20).mean()
        std_vol = df_copy['volume'].rolling(20).std()
        df_copy['volume_zscore'] = (df_copy['volume'] - mean_vol) / (std_vol + 1e-9)
        df_copy['ad_line_ratio'] = ((df_copy['close_smoothed'] - df_copy['low']) - (df_copy['high'] - df_copy['close_smoothed'])) / (df_copy['high'] - df_copy['low'] + 1e-9)
        
        # obv_zscore
        obv_direction = np.where(df_copy['close_smoothed'].diff() > 0, 1, np.where(df_copy['close_smoothed'].diff() < 0, -1, 0))
        obv = (obv_direction * df_copy['volume']).cumsum()
        delta_obv = obv.diff(5)
        std_delta_obv = delta_obv.rolling(20).std()
        df_copy['obv_zscore'] = delta_obv / (std_delta_obv + 1e-9)
        
        pct_change = df_copy['close_smoothed'].pct_change()
        volatility_5d = pct_change.rolling(5).std()
        volatility_60d = pct_change.rolling(60).std()
        df_copy['vol_ratio'] = volatility_5d / (volatility_60d + 1e-9)

        # Nhánh 3: Kỹ thuật, Vĩ mô & Lịch
        df_copy['rsi_14'] = ta.rsi(df_copy['close_smoothed'], length=14)
        
        macd_df = ta.macd(df_copy['close_smoothed'], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            df_copy['macd_ratio'] = macd_df.iloc[:, 0] / (macd_df.iloc[:, 2] + 1e-9)
        else:
            df_copy['macd_ratio'] = 0.0
            
        bb_df = ta.bbands(df_copy['close_smoothed'], length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            df_copy['bb_position'] = (df_copy['close_smoothed'] - bb_df.iloc[:, 0]) / (bb_df.iloc[:, 2] - bb_df.iloc[:, 0] + 1e-9)
        else:
            df_copy['bb_position'] = 0.5
            
        adx_df = ta.adx(df_copy['high'], df_copy['low'], df_copy['close_smoothed'], length=14)
        df_copy['adx_14'] = adx_df.iloc[:, 0] if adx_df is not None else 20.0
        
        stoch_df = ta.stoch(df_copy['high'], df_copy['low'], df_copy['close_smoothed'], fast_k=14)
        df_copy['stoch_k'] = stoch_df.iloc[:, 0] if stoch_df is not None else 50.0
        
        daily_changes = df_copy['close_smoothed'].diff()
        df_copy['efficiency_ratio'] = (df_copy['close_smoothed'] - df_copy['close_smoothed'].shift(10)).abs() / (daily_changes.abs().rolling(10).sum() + 1e-9)

        # Trích xuất và điền khuyết các cột đặc trưng quan tâm
        df_out = pd.DataFrame(index=df.index)
        for col in self.feature_cols:
            if col in df_copy.columns:
                df_out[col] = df_copy[col]
            else:
                df_out[col] = 0.0
            
        # Làm sạch vô cực và NaN lần cuối
        df_out = df_out.replace([np.inf, -np.inf], np.nan)
        for col in self.feature_cols:
            df_out[col] = df_out[col].ffill().bfill().fillna(0.0)
            
        return df_out[self.feature_cols]

    def fit_transform_data(self, df: pd.DataFrame):
        """
        Thực hiện chuẩn hóa dữ liệu sử dụng StandardScaler (mean=0, std=1).
        """
        # Trích xuất đặc trưng dừng
        df_feats = self.transform_df(df)
        
        X_raw = df_feats.values
        y_raw = df[[self.target_col]].values
        
        # Tiến hành học và biến đổi dữ liệu (Scaling)
        X_scaled = self.feature_scaler.fit_transform(X_raw)
        y_scaled = self.target_scaler.fit_transform(y_raw)
        
        y_spread_scaled = None
        if self.spread_col in df.columns:
            y_spread_raw = df[[self.spread_col]].values
            y_spread_scaled = self.spread_scaler.fit_transform(y_spread_raw)
            
        return X_scaled, y_scaled, y_spread_scaled

    def fit_transform_train_only(self, df: pd.DataFrame, train_ratio: float = 0.8, purge_gap: int = 45):
        """
        Huấn luyện bộ chuẩn hóa (fit) CHỈ trên dữ liệu Train và áp dụng (transform) cho toàn bộ dữ liệu.
        Tránh rò rỉ dữ liệu (Data Leakage) từ tập Test sang tập Train.
        """
        # Trích xuất đặc trưng dừng
        df_feats = self.transform_df(df)
        
        # Xác định điểm phân chia (split_idx) tương tự như trong split_train_test_chronological
        total_windows = len(df_feats) - self.time_steps
        split_idx_window = int(total_windows * train_ratio)
        split_idx_raw = split_idx_window + self.time_steps
        
        X_raw = df_feats.values
        y_raw = df[[self.target_col]].values
        
        # Fit bộ chuẩn hóa chỉ trên tập Train (trước split_idx_raw)
        self.feature_scaler.fit(X_raw[:split_idx_raw])
        self.target_scaler.fit(y_raw[:split_idx_raw])
        
        # Transform toàn bộ dữ liệu
        X_scaled = self.feature_scaler.transform(X_raw)
        y_scaled = self.target_scaler.transform(y_raw)
        
        y_spread_scaled = None
        if self.spread_col in df.columns:
            y_spread_raw = df[[self.spread_col]].values
            self.spread_scaler.fit(y_spread_raw[:split_idx_raw])
            y_spread_scaled = self.spread_scaler.transform(y_spread_raw)
            
        return X_scaled, y_scaled, y_spread_scaled



    def create_sliding_windows(self, X_scaled: np.ndarray, y_scaled: np.ndarray, y_spread_scaled: np.ndarray = None):
        """
        Thuật toán Cửa sổ trượt (Sliding Window) để biến đổi dữ liệu thành mảng 3D Tensor.
        """
        X_3D, y_3D, y_spread_3D = [], [], []
        
        # Duyệt qua toàn bộ tập dữ liệu để cắt thành các khối thời gian liên tiếp
        for i in range(self.time_steps, len(X_scaled)):
            # Lấy chuỗi dữ liệu từ ngày (i - time_steps) đến ngày (i - 1) làm đầu vào
            X_3D.append(X_scaled[i - self.time_steps : i])
            # Lấy đáp án của ngày thứ i làm đầu ra để mô hình học
            y_3D.append(y_scaled[i])
            if y_spread_scaled is not None:
                y_spread_3D.append(y_spread_scaled[i])
            
        if y_spread_scaled is not None:
            return np.array(X_3D), np.array(y_3D), np.array(y_spread_3D)
        return np.array(X_3D), np.array(y_3D), None

    def split_train_test_by_year(self, df: pd.DataFrame, X_3D: np.ndarray, y_3D: np.ndarray, y_spread_3D: np.ndarray = None):
        """
        Chia tập dữ liệu theo mốc thời gian lịch sử (theo năm).
        - Giữ nguyên trình tự thời gian (Không dùng shuffle trộn lẫn dữ liệu tương lai).
        """
        df_align = df.iloc[self.time_steps:].reset_index(drop=True)
        df_align['date'] = pd.to_datetime(df_align['date'])
        
        train_mask = df_align['date'].dt.year <= 2023
        test_mask = df_align['date'].dt.year >= 2024
        
        X_train, y_train = X_3D[train_mask], y_3D[train_mask]
        X_test, y_test = X_3D[test_mask], y_3D[test_mask]
        
        y_train_spread = y_spread_3D[train_mask] if y_spread_3D is not None else None
        y_test_spread = y_spread_3D[test_mask] if y_spread_3D is not None else None
        
        y_test_raw = df_align.loc[test_mask, self.target_col].values
        
        return X_train, y_train, X_test, y_test, y_test_raw, y_train_spread, y_test_spread

    def split_train_test_chronological(self, df: pd.DataFrame, X_3D: np.ndarray, y_3D: np.ndarray, y_spread_3D: np.ndarray = None, train_ratio: float = 0.8, purge_gap: int = 45):
        """
        Chia tập dữ liệu theo tỷ lệ thời gian (mặc định 80% Train / 20% Test) kết hợp khoảng cách ly (Purge Gap).
        Giữ nguyên thứ tự thời gian, KHÔNG shuffle.
        """
        total = len(X_3D)
        split_idx = int(total * train_ratio)
        
        X_train = X_3D[:split_idx]
        y_train = y_3D[:split_idx]
        
        test_start = min(split_idx + purge_gap, total)
        X_test  = X_3D[test_start:]
        y_test  = y_3D[test_start:]
        
        y_train_spread = y_spread_3D[:split_idx] if y_spread_3D is not None else None
        y_test_spread = y_spread_3D[test_start:] if y_spread_3D is not None else None
        
        # Lấy mảng target_return thô của tập Test để tính giá thực tế
        df_align = df.iloc[self.time_steps:].reset_index(drop=True)
        y_test_raw = df_align.loc[test_start:, self.target_col].values
        
        print(f"📊 Chia dữ liệu theo tỷ lệ {int(round(train_ratio*100))}/{int(round((1-train_ratio)*100))} (Purge Gap: {purge_gap} phiên):")
        print(f"   🔹 Train: {X_train.shape[0]} mẫu")
        print(f"   🔸 Test : {X_test.shape[0]} mẫu")
        
        return X_train, y_train, X_test, y_test, y_test_raw, y_train_spread, y_test_spread

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Chạy thử nghiệm độc lập Module 2 để soi hình dạng (Shape) của cấu trúc 3D Tensor
    from src.data_loader import fetch_and_prepare_data
    
    try:
        # Cấu hình encoding utf-8 cho Windows console
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
        print("=== KIỂM THỬ PIPELINE BIẾN ĐỔI FEATURE NÂNG CAO ===")
        tickers = ["VNM.VN", "GOOGL", "META"]
        for ticker in tickers:
            print(f"\n==============================")
            print(f"🔬 Đang biến đổi mã: {ticker}")
            print(f"==============================")
            df = fetch_and_prepare_data(ticker, start_date="2010-01-01", end_date="2026-05-20")
            
            # Khởi tạo bộ biến đổi với cửa sổ trượt 30 phiên
            transformer = DataTransformer(time_steps=30)
            X_scaled, y_scaled = transformer.fit_transform_data(df)
            
            # Tạo mảng 3D Tensor
            X_3D, y_3D = transformer.create_sliding_windows(X_scaled, y_scaled)
            print(f"📦 Kích thước mảng 3D (X_3D Shape): {X_3D.shape}")
            
            # Chia tách Train - Test theo thời gian
            X_train, y_train, X_test, y_test, y_test_raw = transformer.split_train_test_chronological(df, X_3D, y_3D, train_ratio=0.8)
            print(f"🔹 Tập Huấn luyện: X_train = {X_train.shape}, y_train = {y_train.shape}")
            print(f"🔸 Tập Kiểm thử   : X_test = {X_test.shape}, y_test = {y_test.shape}")
    except Exception as e:
        print(f"Lỗi kiểm thử features: {e}")