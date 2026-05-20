import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataTransformer:
    def __init__(self, time_steps: int = 30):
        """
        Khởi tạo bộ biến đổi dữ liệu.
        :param time_steps: Số lượng phiên giao dịch quá khứ AI dùng để nhìn lại (mặc định 30 phiên ~ 1.5 tháng).
        """
        self.time_steps = time_steps
        # Khởi tạo 2 bộ scaler độc lập: 1 cho các đặc trưng đầu vào, 1 riêng cho biến mục tiêu (Target)
        # Việc tách riêng scaler giúp sau này chúng ta dễ dàng nghịch đảo (Inverse Transform) để lấy lại giá tiền thực tế
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Định nghĩa các trường dữ liệu đầu vào cho AI
        self.feature_cols = ['close', 'rsi_14', 'MACD_12_26_9', 'volatility_20', 'close_lag1', 'volume_change', 'intraday_return']
        self.target_col = 'target_return' # Đổi tên cột mục tiêu tại đây

    def fit_transform_data(self, df: pd.DataFrame):
        """
        Thực hiện chuẩn hóa dữ liệu về khoảng [0, 1].
        """
        # Trích xuất mảng giá trị thô
        X_raw = df[self.feature_cols].values
        y_raw = df[[self.target_col]].values
        
        # Tiến hành học và biến đổi dữ liệu (Scaling)
        X_scaled = self.feature_scaler.fit_transform(X_raw)
        y_scaled = self.target_scaler.fit_transform(y_raw)
        
        return X_scaled, y_scaled

    def create_sliding_windows(self, X_scaled: np.ndarray, y_scaled: np.ndarray):
        """
        Thuật toán Cửa sổ trượt (Sliding Window) để biến đổi dữ liệu thành mảng 3D Tensor.
        """
        X_3D, y_3D = [], []
        
        # Duyệt qua toàn bộ tập dữ liệu để cắt thành các khối thời gian liên tiếp
        for i in range(self.time_steps, len(X_scaled)):
            # Lấy chuỗi dữ liệu từ ngày (i - time_steps) đến ngày (i - 1) làm đầu vào
            X_3D.append(X_scaled[i - self.time_steps : i])
            # Lấy đáp án của ngày thứ i làm đầu ra để mô hình học
            y_3D.append(y_scaled[i])
            
        return np.array(X_3D), np.array(y_3D)

    def split_train_test_by_year(self, df: pd.DataFrame, X_3D: np.ndarray, y_3D: np.ndarray):
        """
        Workflow Bước 4: Chia tách tập dữ liệu theo mốc thời gian lịch sử chuẩn xác.
        - Giữ nguyên trình tự thời gian (Không dùng shuffle trộn lẫn dữ liệu tương lai).
        """
        # Vì ta cắt mất 'time_steps' dòng đầu tiên khi làm cửa sổ trượt, 
        # nên mảng 3D sẽ khớp dòng bắt đầu từ chỉ số 'time_steps' của DataFrame gốc
        df_align = df.iloc[self.time_steps:].reset_index(drop=True)
        df_align['date'] = pd.to_datetime(df_align['date'])
        
        # Áp dụng chiến lược chia mốc thời gian thực tế của bạn:
        # Train: Toàn bộ dữ liệu trước năm 2024 (Ví dụ từ 2015 - hết 2023)
        # Test/Validate: Dữ liệu từ năm 2024 trở đi (bao gồm cả 2026)
        train_mask = df_align['date'].dt.year <= 2023
        test_mask = df_align['date'].dt.year >= 2024
        
        X_train, y_train = X_3D[train_mask], y_3D[train_mask]
        X_test, y_test = X_3D[test_mask], y_3D[test_mask]
        
        # Lấy ra mảng giá thô của tập Test để sau này so sánh thực tế trên biểu đồ
        y_test_raw = df_align.loc[test_mask, self.target_col].values
        
        return X_train, y_train, X_test, y_test, y_test_raw

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Chạy thử nghiệm độc lập Module 2 để soi hình dạng (Shape) của cấu trúc 3D Tensor
    from src.data_loader import fetch_and_prepare_data
    
    try:
        print("=== KIỂM THỬ PIPELINE BIẾN ĐỔI FEATURE NÂNG CAO ===")
        # 1. Gọi lại dữ liệu sạch từ Module 1
        df = fetch_and_prepare_data("VNM.VN", start_date="2015-01-01", end_date="2026-05-20")
        
        # 2. Khởi tạo bộ biến đổi với cửa sổ trượt 30 phiên
        transformer = DataTransformer(time_steps=30)
        X_scaled, y_scaled = transformer.fit_transform_data(df)
        
        # 3. Tạo mảng 3D Tensor
        X_3D, y_3D = transformer.create_sliding_windows(X_scaled, y_scaled)
        print(f"📦 Kích thước mảng 3D của Tập dữ liệu tổng (X_3D Shape): {X_3D.shape}")
        
        # 4. Chia tách Train - Test theo đúng mốc năm tài chính
        X_train, y_train, X_test, y_test, y_test_raw = transformer.split_train_test_by_year(df, X_3D, y_3D)
        print(f"🔹 Tập Huấn luyện (Train Set) 2015-2023: X_train = {X_train.shape}, y_train = {y_train.shape}")
        print(f"🔸 Tập Kiểm thử (Test Set) 2024-Nay: X_test = {X_test.shape}, y_test = {y_test.shape}")
        
    except Exception as e:
        print(f"💥 Lỗi thực thi Module 2: {e}")