import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf

def fetch_and_prepare_data(ticker: str, start_date: str = "2015-01-01", end_date: str = "2026-05-20"):
    """
    Workflow Bước 1 & 2:
    - Thu thập dữ liệu từ Yahoo Finance (hoặc đọc file CSV dự phòng trong thư mục data).
    - Nếu có file lịch sử cục bộ (VNM_processed.csv), tự động gộp để có bộ dữ liệu dài hơn.
    - Thêm các chỉ báo kỹ thuật (RSI, MACD, Volatility) làm trường dữ liệu mới.
    """
    csv_path = f"data/{ticker}_prices.csv"
    df = None

    # 1. Thu thập dữ liệu (Online từ Yahoo Finance trước, Offline từ CSV sau)
    try:
        print(f"Đang kết nối Yahoo Finance để tải dữ liệu mã: {ticker}...")
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

    # ==========================================
    # NGUỒN DỮ LIỆU CHÍNH: VNM_prices.csv (do trường cung cấp, 2019-2026/03)
    # Đơn vị gốc: nghìn VNĐ → cần nhân 1000 để về VNĐ cho nhất quán với Yahoo Finance
    # ==========================================
    school_data_path = "data/VNM_prices.csv"
    price_cols = ['open', 'high', 'low', 'close']

    if os.path.exists(school_data_path):
        print(f"Dang nap du lieu truong cap: {school_data_path}")
        df_school = pd.read_csv(school_data_path)
        df_school.columns = [str(col).lower() for col in df_school.columns]
        if 'time' in df_school.columns:
            df_school = df_school.rename(columns={'time': 'date'})
        df_school['date'] = pd.to_datetime(df_school['date'])

        # Chuyen don vi: nghin VND → VND (nhan 1000)
        for col in price_cols:
            if col in df_school.columns:
                df_school[col] = df_school[col] * 1000

        school_last_date = df_school['date'].max()
        print(f"  Du lieu truong: {len(df_school)} phien ({df_school['date'].min().date()} → {school_last_date.date()})")

        # Chi lay Yahoo Finance cho cac ngay SAU du lieu truong (tranh trung lap)
        df_yf_new = df[df['date'] > school_last_date].copy()
        if not df_yf_new.empty:
            print(f"  Bu sung Yahoo Finance: {len(df_yf_new)} phien sau {school_last_date.date()}")

        # Cac cot can giu de concat
        keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        school_cols = [c for c in keep_cols if c in df_school.columns]
        yf_cols    = [c for c in keep_cols if c in df_yf_new.columns]

        df_combined = pd.concat([
            df_school[school_cols],
            df_yf_new[yf_cols]
        ], ignore_index=True)
        df_combined = df_combined.sort_values('date').reset_index(drop=True)

        n_tot  = len(df_combined)
        d_from = df_combined['date'].min().date()
        d_to   = df_combined['date'].max().date()
        print(f"  Tong cong: {n_tot} phien ({d_from} → {d_to})")
        df = df_combined
    else:
        # Neu khong co file truong, chi dung Yahoo Finance
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    # 3. Feature Engineering

    print("Khởi động Feature Engineering: Tính toán các chỉ báo kỹ thuật...")

    df['rsi_14'] = ta.rsi(df['close'], length=14)

    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd_df], axis=1)

    # Xóa 2 cột phụ không dùng để tránh noise
    macd_cols_to_drop = [col for col in macd_df.columns if 'MACDh' in col or 'MACDs' in col]
    df = df.drop(columns=macd_cols_to_drop)

    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
    df['close_lag1']    = df['close'].shift(1)
    df['volume_change'] = df['volume'].pct_change()
    df['intraday_return'] = (df['close'] - df['open']) / df['open']

    # Target: tỷ suất lợi nhuận mở cửa ngày mai so với đóng cửa hôm nay
    df['target_return'] = (df['open'].shift(-1) - df['close']) / df['close']

    # Làm sạch NaN
    df_cleaned = df.dropna().reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    cache_output = f"data/{ticker}_processed.csv"
    df_cleaned.to_csv(cache_output, index=False)
    print(f"💾 Đã lưu dữ liệu đặc trưng sạch vào: {cache_output}")
    print(f"📊 Hệ thống sẵn sàng với {len(df_cleaned)} phiên giao dịch hoàn chỉnh.\n")

    return df_cleaned


if __name__ == "__main__":
    try:
        print("=== KIỂM THỬ PIPELINE NẠP DỮ LIỆU CHỈN CHU ===")
        test_df = fetch_and_prepare_data("VNM.VN", start_date="2015-01-01", end_date="2026-05-20")
        print(test_df[['date', 'close', 'rsi_14', 'MACD_12_26_9', 'volatility_20', 'close_lag1', 'volume_change', 'intraday_return', 'target_return']].head())
    except Exception as e:
        print(f"Loi thuc thi Module: {e}")