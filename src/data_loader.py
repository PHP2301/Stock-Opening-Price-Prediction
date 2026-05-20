import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf

def fetch_and_prepare_data(ticker: str, start_date: str = "2015-01-01", end_date: str = "2026-05-20"):
    """
    Workflow Bước 1 & 2:
    - Thu thập dữ liệu từ Yahoo Finance (hoặc đọc file CSV dự phòng trong thư mục data).
    - Thêm các chỉ báo kỹ thuật (RSI, MACD, Volatility) làm trường dữ liệu mới.
    """
    csv_path = f"data/{ticker}_prices.csv"
    df = None

    # 1. Thu thập dữ liệu (Online từ Yahoo Finance trước, Offline từ CSV sau)
    try:
        print(f"Đang kết nối Yahoo Finance để tải dữ liệu mã: {ticker}...")
        # Sử dụng yfinance để cào dữ liệu tự động
        raw_data = yf.download(ticker, start=start_date, end=end_date)
        
        if not raw_data.empty:
            df = raw_data.reset_index()
            print("Tải dữ liệu trực tuyến thành công!")
    except Exception as e:
        print(f"Không thể kết nối Yahoo Finance ({e}). Đang chuyển sang đọc file cục bộ...")

    # Nếu không tải được online, tiến hành tìm file CSV sẵn có trong cây thư mục
    if df is None or df.empty:
        if os.path.exists(csv_path):
            print(f"Tìm thấy file dự phòng: {csv_path}. Đang nạp dữ liệu...")
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Thất bại: Không có kết nối mạng và không tìm thấy file {csv_path}!")

    # 2. Chuẩn hóa cấu trúc bảng dữ liệu ban đầu
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    df = df.sort_values('date').reset_index(drop=True)

    print("Khởi động Feature Engineering: Tính toán các chỉ báo kỹ thuật...")
    
    # Thêm chỉ báo RSI chu kỳ 14
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # Thêm chỉ báo MACD (Lấy cả 3 đường: macd, dòng tín hiệu macds, và cột biến thiên macdh)
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd_df], axis=1)
    
    # Tính độ biến động lịch sử (Volatility) dựa trên độ lệch chuẩn lợi nhuận 20 phiên
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
    
    # Thêm đặc trưng độ trễ (Giá đóng cửa ngày hôm qua)
    df['close_lag1'] = df['close'].shift(1)
    
    # Thêm khối lượng giao dịch đã chuẩn hóa (Tỷ lệ thay đổi khối lượng)
    df['volume_change'] = df['volume'].pct_change()

    # Thêm khoảng chênh lệch giá trong ngày (Intraday Return)
    df['intraday_return'] = (df['close'] - df['open']) / df['open']

    # Tạo biến mục tiêu (Target): Tỷ suất lợi nhuận mở cửa ngày mai so với đóng cửa hôm nay
    # Công thức: (Open_ngày_mai - Close_hôm_nay) / Close_hôm_nay
    df['target_return'] = (df['open'].shift(-1) - df['close']) / df['close']
    
    # Làm sạch triệt để các hàng chứa giá trị NaN (bẫy tính toán thời gian đầu chu kỳ)
    df_cleaned = df.dropna().reset_index(drop=True)
    
    # Đồng bộ cấu hình ghi lại file sạch để làm tài liệu nghiên cứu
    os.makedirs('data', exist_ok=True)
    cache_output = f"data/{ticker}_processed.csv"
    df_cleaned.to_csv(cache_output, index=False)
    print(f"💾 Đã lưu dữ liệu đặc trưng sạch vào: {cache_output}")
    print(f"📊 Hệ thống sẵn sàng với {len(df_cleaned)} phiên giao dịch hoàn chỉnh.\n")
    
    return df_cleaned

if __name__ == "__main__":
    # Khởi chạy kiểm thử độc lập Module 1 với mã Vinamilk trong thư mục của bạn
    # Lưu ý: Trên Yahoo Finance, mã Vinamilk được ký hiệu là VNM.HM (Sàn TP.HCM)
    try:
        print("=== KIỂM THỬ PIPELINE NẠP DỮ LIỆU CHỈN CHU ===")
        test_df = fetch_and_prepare_data("VNM.VN", start_date="2015-01-01", end_date="2026-05-20")
        print(test_df[['date', 'close', 'rsi_14', 'MACD_12_26_9', 'volatility_20', 'close_lag1', 'volume_change', 'intraday_return', 'target_return']].head())
    except Exception as e:
        print(f"💥 Lỗi thực thi Module: {e}")