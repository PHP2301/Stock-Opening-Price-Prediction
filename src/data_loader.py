import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf

def fetch_and_prepare_data(ticker: str, start_date: str = "2015-01-01", end_date: str = "2026-05-20"):
    """
    Workflow Bước 1 & 2:
    - Nguồn chính: VNM_prices.csv (do trường cung cấp, 2019-2026/03, đơn vị nghìn VNĐ).
    - Bổ sung: Yahoo Finance cho các tháng còn thiếu sau ngày cuối của file trường.
    - Thêm các chỉ báo kỹ thuật (RSI, MACD, Volatility).
    """
    school_data_path = "data/VNM_prices.csv"
    price_cols = ['open', 'high', 'low', 'close']

    # ==========================================
    # BƯỚC 1: Tải dữ liệu Yahoo Finance (để lấy phần còn thiếu)
    # ==========================================
    df_yf = None
    try:
        print(f"Dang ket noi Yahoo Finance de tai du lieu ma: {ticker}...")
        raw_data = yf.download(ticker, start=start_date, end=end_date, progress=True)
        if not raw_data.empty:
            df_yf = raw_data.reset_index()
            # Chuan hoa ten cot (xu ly MultiIndex)
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = [str(col[0]).lower() for col in df_yf.columns]
            else:
                df_yf.columns = [str(col).lower() for col in df_yf.columns]
            df_yf['date'] = pd.to_datetime(df_yf['date'])
            df_yf = df_yf.sort_values('date').reset_index(drop=True)
            print(f"  Yahoo Finance: {len(df_yf)} phien ({df_yf['date'].min().date()} → {df_yf['date'].max().date()})")
    except Exception as e:
        print(f"Khong the ket noi Yahoo Finance: {e}")

    # ==========================================
    # BƯỚC 2: Load dữ liệu trường + nhân 1000 đổi đơn vị
    # VNM_prices.csv đơn vị: nghìn VNĐ → nhân 1000 → VNĐ (khớp với Yahoo Finance)
    # ==========================================
    if not os.path.exists(school_data_path):
        if df_yf is None or df_yf.empty:
            raise FileNotFoundError("Khong co du lieu truong va khong ket noi duoc Yahoo Finance!")
        print("Khong tim thay VNM_prices.csv, chi dung Yahoo Finance.")
        df = df_yf.copy()
    else:
        print(f"Dang nap du lieu truong: {school_data_path}")
        df_school = pd.read_csv(school_data_path)
        df_school.columns = [str(col).lower() for col in df_school.columns]
        # VNM_prices.csv dung cot 'time' thay vi 'date'
        if 'time' in df_school.columns:
            df_school = df_school.rename(columns={'time': 'date'})
        df_school['date'] = pd.to_datetime(df_school['date'])
        df_school = df_school.sort_values('date').reset_index(drop=True)

        # Nhan 1000: chuyen tu nghin VND sang VND
        for col in price_cols:
            if col in df_school.columns:
                df_school[col] = df_school[col] * 1000

        school_last_date = df_school['date'].max()
        print(f"  Du lieu truong: {len(df_school)} phien ({df_school['date'].min().date()} → {school_last_date.date()})")
        print(f"  Gia cuoi: {df_school.iloc[-1]['close']:,.0f} VND (sau * 1000)")

        # Lay Yahoo Finance chi cho cac ngay SAU du lieu truong
        keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df_yf_new = pd.DataFrame()
        if df_yf is not None and not df_yf.empty:
            df_yf_new = df_yf[df_yf['date'] > school_last_date].copy()
            if not df_yf_new.empty:
                print(f"  Bu sung Yahoo: {len(df_yf_new)} phien sau {school_last_date.date()}")

        # Gop 2 nguon
        school_keep = [c for c in keep_cols if c in df_school.columns]
        yf_keep     = [c for c in keep_cols if c in df_yf_new.columns]

        frames = [df_school[school_keep]]
        if not df_yf_new.empty:
            frames.append(df_yf_new[yf_keep])

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  Tong cong: {len(df)} phien ({df['date'].min().date()} → {df['date'].max().date()})")

    # ==========================================
    # BƯỚC 3: Feature Engineering
    # ==========================================
    print("Khoi dong Feature Engineering: Tinh toan cac chi bao ky thuat...")

    df['rsi_14'] = ta.rsi(df['close'], length=14)

    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd_df], axis=1)
    macd_cols_to_drop = [col for col in macd_df.columns if 'MACDh' in col or 'MACDs' in col]
    df = df.drop(columns=macd_cols_to_drop)

    df['volatility_20']   = df['close'].pct_change().rolling(window=20).std()
    df['close_lag1']      = df['close'].shift(1)
    df['volume_change']   = df['volume'].pct_change()
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Tinh toan Bollinger Bands (20, 2) va đặt tên cột tường minh
    bb_df = ta.bbands(df['close'], length=20, std=2)
    df['bb_lower'] = bb_df.iloc[:, 0]
    df['bb_middle'] = bb_df.iloc[:, 1]
    df['bb_upper'] = bb_df.iloc[:, 2]

    # Tinh toan ATR (14)
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Biến mục tiêu: tỷ suất lợi nhuận mở cửa ngày mai so với đóng cửa hôm nay
    df['target_return']   = (df['open'].shift(-1) - df['close']) / df['close']

    df_cleaned = df.dropna().reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    cache_output = f"data/{ticker}_processed.csv"
    df_cleaned.to_csv(cache_output, index=False)
    print(f"Da luu du lieu dac trung sach vao: {cache_output}")
    print(f"He thong san sang voi {len(df_cleaned)} phien giao dich hoan chinh.\n")

    return df_cleaned


if __name__ == "__main__":
    try:
        print("=== KIEM THU PIPELINE NAP DU LIEU ===")
        test_df = fetch_and_prepare_data("VNM.VN", start_date="2015-01-01", end_date="2026-05-21")
        cols_show = ['date', 'close', 'rsi_14', 'MACD_12_26_9', 'target_return']
        print("\n--- 3 phien dau (2019) ---")
        print(test_df[cols_show].head(3).to_string(index=False))
        print("\n--- 3 phien cuoi (2026) ---")
        print(test_df[cols_show].tail(3).to_string(index=False))
    except Exception as e:
        print(f"Loi: {e}")