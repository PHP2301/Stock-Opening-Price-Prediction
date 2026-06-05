import numpy as np
import pandas as pd
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
    def __init__(self, time_steps: int = 30):
        """
        Khởi tạo bộ biến đổi dữ liệu.
        :param time_steps: Số lượng phiên giao dịch quá khứ AI dùng để nhìn lại (mặc định 30 phiên ~ 1.5 tháng).
        """
        self.time_steps = time_steps
        # Khởi tạo 2 bộ scaler độc lập: 1 cho các đặc trưng đầu vào, 1 riêng cho biến mục tiêu (Target)
        # Sử dụng StandardScaler để tránh bị ảnh hưởng bởi các giá trị ngoại lai (outliers)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.spread_scaler = StandardScaler()
        
        # Định nghĩa các trường dữ liệu đầu vào dừng cho AI (24 đặc trưng)
        self.feature_cols = [
            'rsi_14', 'macd_ratio', 'volatility_20', 
            'close_lag1_ratio', 'close_lag2_ratio', 'close_lag3_ratio',
            'open_lag1_ratio', 'open_lag2_ratio', 'rsi_lag1',
            'volume_change', 'intraday_return',
            'bb_lower_ratio', 'bb_middle_ratio', 'bb_upper_ratio', 'atr_ratio',
            'ema_14_ratio', 'roc_10', 'adx_14', 'market_return', 'vix',
            'sentiment_score', 'news_volume', 'bond_yield_10y', 'dollar_index_change'
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
        
        # Chuyển đổi các đặc trưng giá tuyệt đối sang dạng tỷ lệ so với close ngày hiện tại
        df_copy['ema_14_ratio'] = (df_copy['ema_14'] - df_copy['close']) / df_copy['close']
        df_copy['bb_lower_ratio'] = (df_copy['bb_lower'] - df_copy['close']) / df_copy['close']
        df_copy['bb_middle_ratio'] = (df_copy['bb_middle'] - df_copy['close']) / df_copy['close']
        df_copy['bb_upper_ratio'] = (df_copy['bb_upper'] - df_copy['close']) / df_copy['close']
        df_copy['atr_ratio'] = df_copy['atr_14'] / df_copy['close']
        
        # Tương thích cột MACD
        macd_cols = [c for c in df_copy.columns if 'MACD_12_26_9' in c or 'macd' in c.lower()]
        if macd_cols:
            df_copy['macd_ratio'] = df_copy[macd_cols[0]] / df_copy['close']
        else:
            df_copy['macd_ratio'] = 0.0
            
        df_copy['close_lag1_ratio'] = (df_copy['close_lag1'] - df_copy['close']) / df_copy['close']
        df_copy['close_lag2_ratio'] = (df_copy['close_lag2'] - df_copy['close']) / df_copy['close']
        df_copy['close_lag3_ratio'] = (df_copy['close_lag3'] - df_copy['close']) / df_copy['close']
        df_copy['open_lag1_ratio'] = (df_copy['open_lag1'] - df_copy['close']) / df_copy['close']
        df_copy['open_lag2_ratio'] = (df_copy['open_lag2'] - df_copy['close']) / df_copy['close']
        
        # Tương thích tên chỉ số vix
        vix_cols = [c for c in df_copy.columns if c.lower() in ['vix', 'vix_close']]
        if vix_cols:
            df_copy['vix'] = df_copy[vix_cols[0]]
        else:
            df_copy['vix'] = 20.0
            
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

    def split_train_test_chronological(self, df: pd.DataFrame, X_3D: np.ndarray, y_3D: np.ndarray, y_spread_3D: np.ndarray = None, train_ratio: float = 0.8):
        """
        Chia tập dữ liệu theo tỷ lệ thời gian (mặc định 80% Train / 20% Test).
        Phù hợp khi dữ liệu lịch sử bị giới hạn (ví dụ: Yahoo Finance chỉ có từ 2023).
        Giữ nguyên thứ tự thời gian, KHÔNG shuffle.
        """
        total = len(X_3D)
        split_idx = int(total * train_ratio)
        
        X_train = X_3D[:split_idx]
        y_train = y_3D[:split_idx]
        X_test  = X_3D[split_idx:]
        y_test  = y_3D[split_idx:]
        
        y_train_spread = y_spread_3D[:split_idx] if y_spread_3D is not None else None
        y_test_spread = y_spread_3D[split_idx:] if y_spread_3D is not None else None
        
        # Lấy mảng target_return thô của tập Test để tính giá thực tế
        df_align = df.iloc[self.time_steps:].reset_index(drop=True)
        y_test_raw = df_align.loc[split_idx:, self.target_col].values
        
        print(f"📊 Chia dữ liệu theo tỷ lệ {int(round(train_ratio*100))}/{int(round((1-train_ratio)*100))}:")
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