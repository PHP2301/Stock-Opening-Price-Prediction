import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf

def format_vn(value):
    if value is None:
        return ""
    # Chuyển đổi định dạng số kiểu Anh (phẩy hàng nghìn, chấm thập phân) sang kiểu Việt (chấm hàng nghìn, phẩy thập phân)
    s = f"{value:,.2f}"
    temp = s.replace(",", "TEMP")
    temp = temp.replace(".", ",")
    temp = temp.replace("TEMP", ".")
    return temp

def fetch_and_prepare_data(ticker: str, start_date: str = "2015-01-01", end_date: str = "2026-05-20"):
    """
    Workflow Bước 1 & 2:
    - Nguồn chính: VNM_prices.csv (do trường cung cấp, 2019-2026/03, đơn vị nghìn VNĐ).
    - Bổ sung: Yahoo Finance cho các tháng còn thiếu sau ngày cuối của file trường.
    - Thêm các chỉ báo kỹ thuật (RSI, MACD, Volatility).
    """
    # Tu dong xac dinh duong dan file truong cung cap
    base_ticker = ticker.split('.')[0]
    school_data_path = f"data/{base_ticker}_prices.csv"
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
            
            # Đổi giá USD sang VNĐ cho Yahoo Finance
            if "VNM" not in ticker.upper():
                USD_TO_VND = 25400
                for col in price_cols:
                    if col in df_yf.columns:
                        df_yf[col] = df_yf[col] * USD_TO_VND
                print(f"  [QUY ĐỔI] Đã quy đổi giá trị Yahoo Finance {ticker} sang VNĐ (tỷ giá 1 USD = {format_vn(USD_TO_VND)} VNĐ)")
                
            print(f"  Yahoo Finance: {len(df_yf)} phien ({df_yf['date'].min().date()} to {df_yf['date'].max().date()})")
    except Exception as e:
        print(f"Khong the ket noi Yahoo Finance: {e}")

    # ==========================================
    # BƯỚC 2: Load dữ liệu trường + quy đổi đơn vị
    # ==========================================
    if not os.path.exists(school_data_path):
        if df_yf is None or df_yf.empty:
            raise FileNotFoundError(f"Khong co du lieu truong {school_data_path} va khong ket noi duoc Yahoo Finance!")
        print(f"Khong tim thay {school_data_path}, chi dung Yahoo Finance.")
        df = df_yf.copy()
    else:
        print(f"Dang nap du lieu truong: {school_data_path}")
        df_school = pd.read_csv(school_data_path)
        df_school.columns = [str(col).lower() for col in df_school.columns]
        # VNM_prices.csv va cac file khac co the dung cot 'time' hoac 'date'
        if 'time' in df_school.columns:
            df_school = df_school.rename(columns={'time': 'date'})
        df_school['date'] = pd.to_datetime(df_school['date'])
        df_school = df_school.sort_values('date').reset_index(drop=True)

        # Tải dữ liệu lịch sử VNM từ DNSE cho các năm trước dữ liệu trường (trước 2019-09-17)
        df_dnse = pd.DataFrame()
        if "VNM" in ticker.upper():
            try:
                import urllib.request
                import json
                
                school_first_date = df_school['date'].min()
                # Chuyển đổi sang epoch timestamp
                start_epoch = int(pd.to_datetime(start_date).timestamp())
                end_epoch = int(school_first_date.timestamp())
                
                print(f"  Dang tai bo sung du lieu {ticker} truoc 2019 tu DNSE Chart API...")
                url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from={start_epoch}&to={end_epoch}&symbol=VNM&resolution=1D"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as res:
                    dnse_raw = json.loads(res.read())
                
                if 't' in dnse_raw and len(dnse_raw['t']) > 0:
                    df_dnse = pd.DataFrame({
                        'date': pd.to_datetime(dnse_raw['t'], unit='s'),
                        'open': dnse_raw['o'],
                        'high': dnse_raw['h'],
                        'low': dnse_raw['l'],
                        'close': dnse_raw['c'],
                        'volume': dnse_raw['v']
                    })
                    # Lọc lấy các ngày trước ngày bắt đầu của dữ liệu trường
                    df_dnse = df_dnse[df_dnse['date'] < school_first_date]
                    print(f"  Tai bo sung DNSE thanh cong: {len(df_dnse)} phien tu {df_dnse['date'].min().date()} den {df_dnse['date'].max().date()}")
            except Exception as e:
                print(f"  Khong the tai du lieu tu DNSE: {e}")

        # Đổi đơn vị sang VNĐ cho dữ liệu trường và dữ liệu DNSE bổ sung
        if "VNM" in ticker.upper():
            for col in price_cols:
                if col in df_school.columns:
                    df_school[col] = df_school[col] * 1000
                if not df_dnse.empty and col in df_dnse.columns:
                    df_dnse[col] = df_dnse[col] * 1000
        else:
            USD_TO_VND = 25400
            for col in price_cols:
                if col in df_school.columns:
                    df_school[col] = df_school[col] * USD_TO_VND
            print(f"  [QUY ĐỔI] Đã quy đổi giá trị dữ liệu trường {ticker} sang VNĐ (tỷ giá 1 USD = {format_vn(USD_TO_VND)} VNĐ)")

        school_last_date = df_school['date'].max()
        print(f"  Du lieu truong: {len(df_school)} phien ({df_school['date'].min().date()} to {school_last_date.date()})")
        print(f"  Gia cuoi: {format_vn(df_school.iloc[-1]['close'])} (sau tien xu ly)")

        # Lay Yahoo Finance chi cho cac ngay SAU du lieu truong
        keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df_yf_new = pd.DataFrame()
        df_yf_before = pd.DataFrame()
        if df_yf is not None and not df_yf.empty:
            school_first_date = df_school['date'].min()
            df_yf_before = df_yf[df_yf['date'] < school_first_date].copy()
            if not df_yf_before.empty:
                print(f"  Bu sung Yahoo (truoc 2020): {len(df_yf_before)} phien truoc {school_first_date.date()}")
                
            df_yf_new = df_yf[df_yf['date'] > school_last_date].copy()
            if not df_yf_new.empty:
                print(f"  Bu sung Yahoo: {len(df_yf_new)} phien sau {school_last_date.date()}")

        # Gop cac nguon
        school_keep = [c for c in keep_cols if c in df_school.columns]
        yf_keep     = [c for c in keep_cols if c in df_yf_new.columns]

        frames = []
        if not df_dnse.empty:
            frames.append(df_dnse[keep_cols])
        elif not df_yf_before.empty:
            frames.append(df_yf_before[keep_cols])
            
        frames.append(df_school[school_keep])
        if not df_yf_new.empty:
            frames.append(df_yf_new[yf_keep])

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  Tong cong: {len(df)} phien ({df['date'].min().date()} to {df['date'].max().date()})")

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

    # Tinh toan bo sung EMA (14), ROC (10), ADX (14)
    df['ema_14'] = ta.ema(df['close'], length=14)
    df['roc_10'] = ta.roc(df['close'], length=10)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx_14'] = adx_df.iloc[:, 0]

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
    import sys
    try:
        # Cấu hình encoding utf-8 cho Windows console
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
        # Cấu hình định dạng hiển thị của pandas để không bị số mũ khoa học (scientific notation)
        pd.options.display.float_format = '{:,.2f}'.format
        
        print("=== KIEM THU PIPELINE NAP DU LIEU ===")
        tickers = ["VNM.VN", "GOOGL", "META"]
        for ticker in tickers:
            print(f"\n==============================")
            print(f"🔬 Đang tải thử mã: {ticker}")
            print(f"==============================")
            test_df = fetch_and_prepare_data(ticker, start_date="2010-01-01", end_date="2026-05-21")
            cols_show = ['date', 'close', 'rsi_14', 'MACD_12_26_9', 'target_return']
            print("\n--- 3 phien dau ---")
            print(test_df[cols_show].head(3).to_string(index=False))
            print("\n--- 3 phien cuoi ---")
            print(test_df[cols_show].tail(3).to_string(index=False))
    except Exception as e:
        print(f"Loi kiem thu: {e}")