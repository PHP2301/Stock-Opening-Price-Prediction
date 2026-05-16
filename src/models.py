import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

class StockPredictor:
    def __init__(self, df):
        """
        Khởi tạo class với dữ liệu đã được làm sạch và trích xuất đặc trưng
        """
        self.df = df.copy()
        # Định nghĩa các biến đầu vào (Features) và biến mục tiêu (Target)
        self.features = ['close', 'RSI_14', 'MACD_12_26_9', 'Volatility_20', 'Close_Lag1']
        self.target = 'Next_Open'
        
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        
        # Chia dữ liệu: 80% để học (Train), 20% để kiểm tra (Test)
        # Vì là chuỗi thời gian, ta KHÔNG dùng shuffle=True để tránh rò rỉ dữ liệu tương lai
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=False
        )

    def train_linear_regression(self):
        """Huấn luyện mô hình Baseline: Tuyến tính"""
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        return model, self._evaluate("Linear Regression", predictions)

    def train_xgboost(self):
        """Huấn luyện mô hình nâng cao: XGBoost"""
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        return model, self._evaluate("XGBoost", predictions)

    def _evaluate(self, model_name, preds):
        """Thuật toán tính toán sai số để đánh giá mô hình"""
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        mae = mean_absolute_error(self.y_test, preds)
        print(f"==={model_name} ===")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        return {"RMSE": rmse, "MAE": mae, "Predictions": preds}