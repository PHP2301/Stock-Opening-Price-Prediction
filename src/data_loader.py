import pandas as pd
import pandas_ta as ta

def load_stock_data(file_path):
    # Bước 1: Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Bước 2: Chuẩn hóa tên cột
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Bước 3: Feature Engineering (Tính chỉ báo kỹ thuật)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd_df], axis=1)
    df['Volatility_20'] = df['close'].pct_change().rolling(window=20).std()
    
    # Bước 4: Tạo Lag Features (Biến độ trễ)
    df['Close_Lag1'] = df['close'].shift(1)
    
    # Bước 5: Tạo biến mục tiêu (Giá mở cửa ngày mai)
    df['Next_Open'] = df['open'].shift(-1)
    
    # Xóa bỏ các dòng NaN do shift và chỉ báo tạo ra để sạch dữ liệu
    df = df.dropna()
    
    return df