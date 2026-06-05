import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np

def format_vn(value):
    if value is None:
        return ""
    # Chuyển đổi định dạng số kiểu Anh (phẩy hàng nghìn, chấm thập phân) sang kiểu Việt (chấm hàng nghìn, phẩy thập phân)
    s = f"{value:,.2f}"
    temp = s.replace(",", "TEMP")
    temp = temp.replace(".", ",")
    temp = temp.replace("TEMP", ".")
    return temp

def calculate_days_before_tet(dates_series):
    """
    Tính toán số phiên giao dịch thực tế còn lại đến Tết Nguyên Đán (cáp tối đa 30 phiên).
    """
    tet_dates = pd.to_datetime([
        "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31",
        "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
        "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
        "2025-01-29", "2026-02-17"
    ])
    days_before = []
    dates_list = pd.to_datetime(dates_series).tolist()
    
    tet_last_trading_days = []
    for tet in tet_dates:
        before_tet = [d for d in dates_list if d < tet]
        if before_tet:
            tet_last_trading_days.append(max(before_tet))
        else:
            tet_last_trading_days.append(None)
            
    for d in dates_list:
        next_tets = [t for t in tet_last_trading_days if t is not None and t >= d]
        if next_tets:
            next_tet_last_trade = min(next_tets)
            idx_d = dates_list.index(d)
            idx_tet = dates_list.index(next_tet_last_trade)
            diff = idx_tet - idx_d
            days_before.append(min(diff, 30))
        else:
            days_before.append(30)
    return days_before

def get_realtime_usd_vnd_rate():
    """Tải tỷ giá USD/VND trực tuyến (realtime) từ Yahoo Finance và API công cộng."""
    # Thử yfinance trước
    try:
        ticker = yf.Ticker("USDVND=X")
        df = ticker.history(period="1d")
        if not df.empty:
            rate = float(df['Close'].iloc[-1])
            # Chuẩn hóa nếu < 1000
            if rate < 1000.0:
                rate *= 1000.0
            if 15000.0 <= rate <= 28000.0:
                return rate
    except Exception:
        pass

    # Thử API công cộng làm dự phòng
    try:
        import urllib.request
        import json
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            rate = float(data['rates']['VND'])
            if 15000.0 <= rate <= 28000.0:
                return rate
    except Exception:
        pass

    # Fallback mặc định nếu mất kết nối mạng hoàn toàn
    return 25400.0

def log_feature_correlation_comparison(df, ticker):
    """
    Tính toán và in ra hệ số tương quan Pearson giữa các đặc trưng kỹ thuật với target_return
    trước và sau khi áp dụng Kalman Filter để xác định mức độ cải thiện tín hiệu.
    """
    raw_close = df['close']
    smoothed_close = df['close_smoothed']
    target_return = df['target_return']
    
    # Danh sách các chỉ báo cần tính toán và so sánh tương quan
    indicators = [
        ('RSI_14', lambda p: ta.rsi(p, length=14)),
        ('MACD', lambda p: ta.macd(p, fast=12, slow=26, signal=9).iloc[:, 0] if ta.macd(p, fast=12, slow=26, signal=9) is not None else None),
        ('Volatility_20', lambda p: p.pct_change().rolling(window=20).std()),
        ('EMA_14', lambda p: ta.ema(p, length=14)),
        ('ROC_10', lambda p: ta.roc(p, length=10)),
        ('Lag_1', lambda p: p.shift(1)),
        ('Lag_2', lambda p: p.shift(2)),
        ('Lag_3', lambda p: p.shift(3))
    ]
    
    print("\n=================================================================================")
    print(f"📊 PHÂN TÍCH TƯƠNG QUAN ĐẶC TRƯNG VỚI TARGET (TICKER: {ticker})")
    print(f"   (So sánh trước vs sau khi áp dụng Kalman Filter để làm mịn giá)")
    print("=================================================================================")
    print(f" {'Tên đặc trưng':<18} | {'Tương quan Thô':<16} | {'Tương quan Kalman':<18} | {'Tăng/Giảm hiệu suất':<20}")
    print("-" * 81)
    
    for name, func in indicators:
        try:
            feat_raw = func(raw_close)
            feat_smooth = func(smoothed_close)
            if feat_raw is None or feat_smooth is None:
                continue
            
            temp_df = pd.DataFrame({
                'raw': feat_raw,
                'smooth': feat_smooth,
                'target': target_return
            }).dropna()
            
            corr_raw = temp_df['raw'].corr(temp_df['target'])
            corr_smooth = temp_df['smooth'].corr(temp_df['target'])
            
            # Cải thiện là khi tương quan trị tuyệt đối tăng lên (tín hiệu mạnh hơn)
            diff = abs(corr_smooth) - abs(corr_raw)
            if np.isnan(diff):
                diff_str = "N/A"
            else:
                diff_str = f"{diff:+.6f} ({'Cải thiện' if diff > 0 else 'Giảm tín hiệu'})"
            
            print(f" {name:<18} | {corr_raw:>16.6f} | {corr_smooth:>18.6f} | {diff_str:<20}")
        except Exception as e:
            print(f" {name:<18} | Lỗi tính toán: {e}")
    print("=================================================================================\n")

def fetch_and_prepare_data(ticker: str, start_date: str = "2015-01-01", end_date: str = "2026-05-20", sentiment_engine: str = 'vader'):
    """
    Workflow Bước 1 & 2:
    - Nguồn chính: VNM_prices.csv (do trường cung cấp, 2019-2026/03, đơn vị nghìn VNĐ).
    - Bổ sung: Yahoo Finance cho các tháng còn thiếu sau ngày cuối của file trường.
    - Thêm các chỉ báo kỹ thuật (RSI, MACD, Volatility).
    """
    # Xác định đường dẫn tuyệt đối đến thư mục gốc của dự án
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    # Tu dong xac dinh duong dan file truong cung cap
    base_ticker = ticker.split('.')[0]
    school_data_path = os.path.join(data_dir, "raw", f"{base_ticker}_prices.csv")
    price_cols = ['open', 'high', 'low', 'close']

    # Tải tỷ giá USD/VND động trực tuyến từ Yahoo Finance
    df_usd_vnd = None
    try:
        print(f"Dang tai ty gia USD/VND truc tuyen (USDVND=X) tu {start_date} den {end_date}...")
        raw_rate = yf.download("USDVND=X", start=start_date, end=end_date, progress=False)
        if not raw_rate.empty:
            df_usd_vnd = raw_rate.reset_index()
            if isinstance(df_usd_vnd.columns, pd.MultiIndex):
                df_usd_vnd.columns = [str(col[0]).lower() for col in df_usd_vnd.columns]
            else:
                df_usd_vnd.columns = [str(col).lower() for col in df_usd_vnd.columns]
            df_usd_vnd['date'] = pd.to_datetime(df_usd_vnd['date'])
            df_usd_vnd = df_usd_vnd[['date', 'close']].rename(columns={'close': 'rate_close'})
            df_usd_vnd = df_usd_vnd.sort_values('date').reset_index(drop=True)
            # Chuẩn hóa tỷ giá: nếu tỷ giá < 1000 (do yfinance lưu dạng 19.5 thay vì 19500), nhân với 1000
            df_usd_vnd['rate_close'] = df_usd_vnd['rate_close'].apply(lambda x: x * 1000.0 if x < 1000.0 else x)
            # Loại bỏ các giá trị nhiễu ngoài khoảng tỷ giá USD/VND lịch sử hợp lý [15000, 28000]
            df_usd_vnd.loc[(df_usd_vnd['rate_close'] < 15000.0) | (df_usd_vnd['rate_close'] > 28000.0), 'rate_close'] = np.nan
            df_usd_vnd['rate_close'] = df_usd_vnd['rate_close'].ffill().bfill().fillna(get_realtime_usd_vnd_rate())
    except Exception as e:
        print(f"  [WARNING] Khong the tai ty gia truc tuyen USDVND=X: {e}")

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
                if df_usd_vnd is not None and not df_usd_vnd.empty:
                    df_yf = pd.merge(df_yf, df_usd_vnd, on='date', how='left')
                    df_yf['rate_close'] = df_yf['rate_close'].ffill().bfill().fillna(25400.0)
                    for col in price_cols:
                        if col in df_yf.columns:
                            df_yf[col] = df_yf[col] * df_yf['rate_close']
                    df_yf = df_yf.drop(columns=['rate_close'])
                    print(f"  [QUY ĐỔI] Đã quy đổi giá trị Yahoo Finance {ticker} sang VNĐ bằng tỷ giá động trực tuyến.")
                else:
                    USD_TO_VND = 25400
                    for col in price_cols:
                        if col in df_yf.columns:
                            df_yf[col] = df_yf[col] * USD_TO_VND
                    print(f"  [QUY ĐỔI] Đã quy đổi giá trị Yahoo Finance {ticker} sang VNĐ (tỷ giá 1 USD = {format_vn(USD_TO_VND)} VNĐ - Fallback)")
                
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
            if df_usd_vnd is not None and not df_usd_vnd.empty:
                df_school = pd.merge(df_school, df_usd_vnd, on='date', how='left')
                df_school['rate_close'] = df_school['rate_close'].ffill().bfill().fillna(25400.0)
                for col in price_cols:
                    if col in df_school.columns:
                        df_school[col] = df_school[col] * df_school['rate_close']
                df_school = df_school.drop(columns=['rate_close'])
                print(f"  [QUY ĐỔI] Đã quy đổi giá trị dữ liệu trường {ticker} sang VNĐ bằng tỷ giá động trực tuyến.")
            else:
                USD_TO_VND = 25400
                for col in price_cols:
                    if col in df_school.columns:
                        df_school[col] = df_school[col] * USD_TO_VND
                print(f"  [QUY ĐỔI] Đã quy đổi giá trị dữ liệu trường {ticker} sang VNĐ (tỷ giá 1 USD = {format_vn(USD_TO_VND)} VNĐ - Fallback)")

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

    # Tải chỉ số thị trường tương ứng làm tham chiếu vĩ mô
    is_vn = ".VN" in ticker.upper()
    index_ticker = "VNM" if is_vn else "^IXIC"
    try:
        print(f"  [MARKET] Dang tai chi so thi truong {index_ticker} tu Yahoo Finance...")
        raw_index = yf.download(index_ticker, start=start_date, end=end_date, progress=False)
        if not raw_index.empty:
            df_index = raw_index.reset_index()
            if isinstance(df_index.columns, pd.MultiIndex):
                df_index.columns = [str(col[0]).lower() for col in df_index.columns]
            else:
                df_index.columns = [str(col).lower() for col in df_index.columns]
            df_index['date'] = pd.to_datetime(df_index['date'])
            df_index['market_return'] = df_index['close'].pct_change()
            df_index = df_index[['date', 'market_return']]
        else:
            df_index = pd.DataFrame(columns=['date', 'market_return'])
    except Exception as e:
        print(f"  [WARNING] Khong the tai du lieu chi so {index_ticker}: {e}")
        df_index = pd.DataFrame(columns=['date', 'market_return'])

    # Ghep dac trung market_return vao DataFrame goc theo date
    df = pd.merge(df, df_index, on='date', how='left')
    df['market_return'] = df['market_return'].fillna(0.0)

    # Tải chỉ số hoảng sợ vĩ mô VIX (^VIX)
    try:
        print(f"  [VIX] Dang tai chi so so hai VIX (^VIX) tu Yahoo Finance...")
        raw_vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if not raw_vix.empty:
            df_vix = raw_vix.reset_index()
            if isinstance(df_vix.columns, pd.MultiIndex):
                df_vix.columns = [str(col[0]).lower() for col in df_vix.columns]
            else:
                df_vix.columns = [str(col).lower() for col in df_vix.columns]
            df_vix['date'] = pd.to_datetime(df_vix['date'])
            df_vix = df_vix[['date', 'close']].rename(columns={'close': 'vix'})
        else:
            df_vix = pd.DataFrame(columns=['date', 'vix'])
    except Exception as e:
        print(f"  [WARNING] Khong the tai chi so VIX: {e}")
        df_vix = pd.DataFrame(columns=['date', 'vix'])

    # Ghep dac trung vix vao DataFrame goc theo date
    df = pd.merge(df, df_vix, on='date', how='left')
    df['vix'] = df['vix'].ffill().bfill().fillna(20.0)

    # Tải Chỉ số Lợi suất trái phiếu chính phủ Mỹ 10 năm (^TNX)
    try:
        print(f"  [MACRO] Dang tai chi so loi suat trai phieu ^TNX tu Yahoo Finance...")
        raw_tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)
        if not raw_tnx.empty:
            df_tnx = raw_tnx.reset_index()
            if isinstance(df_tnx.columns, pd.MultiIndex):
                df_tnx.columns = [str(col[0]).lower() for col in df_tnx.columns]
            else:
                df_tnx.columns = [str(col).lower() for col in df_tnx.columns]
            df_tnx['date'] = pd.to_datetime(df_tnx['date'])
            df_tnx = df_tnx[['date', 'close']].rename(columns={'close': 'bond_yield_10y'})
        else:
            df_tnx = pd.DataFrame(columns=['date', 'bond_yield_10y'])
    except Exception as e:
        print(f"  [WARNING] Khong the tai ^TNX: {e}")
        df_tnx = pd.DataFrame(columns=['date', 'bond_yield_10y'])

    df = pd.merge(df, df_tnx, on='date', how='left')
    df['bond_yield_10y'] = df['bond_yield_10y'].ffill().bfill().fillna(4.0)

    # Tải Chỉ số Dollar Index (DX-Y.NYB)
    try:
        print(f"  [MACRO] Dang tai chi so Dollar Index DX-Y.NYB tu Yahoo Finance...")
        raw_dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False)
        if not raw_dxy.empty:
            df_dxy = raw_dxy.reset_index()
            if isinstance(df_dxy.columns, pd.MultiIndex):
                df_dxy.columns = [str(col[0]).lower() for col in df_dxy.columns]
            else:
                df_dxy.columns = [str(col).lower() for col in df_dxy.columns]
            df_dxy['date'] = pd.to_datetime(df_dxy['date'])
            df_dxy['dollar_index_change'] = df_dxy['close'].pct_change()
            df_dxy = df_dxy[['date', 'dollar_index_change']]
        else:
            df_dxy = pd.DataFrame(columns=['date', 'dollar_index_change'])
    except Exception as e:
        print(f"  [WARNING] Khong the tai DX-Y.NYB: {e}")
        df_dxy = pd.DataFrame(columns=['date', 'dollar_index_change'])

    df = pd.merge(df, df_dxy, on='date', how='left')
    df['dollar_index_change'] = df['dollar_index_change'].fillna(0.0)

    # Ghep usdvnd_change tu df_usd_vnd
    if df_usd_vnd is not None and not df_usd_vnd.empty:
        df_usd_vnd['usdvnd_change'] = df_usd_vnd['rate_close'].pct_change()
        df = pd.merge(df, df_usd_vnd[['date', 'usdvnd_change']], on='date', how='left')
    else:
        df['usdvnd_change'] = 0.0
    df['usdvnd_change'] = df['usdvnd_change'].fillna(0.0)

    # Đồng bộ múi giờ (Timezone Alignment)
    if is_vn:
        df['vix_lag1'] = df['vix'].shift(1)
        df['bond_yield_lag1'] = df['bond_yield_10y'].shift(1)
        df['usdvnd_change'] = df['usdvnd_change'].shift(1)
        df['vnindex_return_lag1'] = df['market_return'].shift(1)  # VNM ETF dịch trễ 1 ngày vì lệch múi giờ Mỹ
    else:
        df['vix_lag1'] = df['vix']
        df['bond_yield_lag1'] = df['bond_yield_10y']
        df['usdvnd_change'] = df['dollar_index_change']  # DXY cho US stock
        df['vnindex_return_lag1'] = df['market_return']

    df['vix_lag1'] = df['vix_lag1'].ffill().bfill().fillna(20.0)
    df['bond_yield_lag1'] = df['bond_yield_lag1'].ffill().bfill().fillna(4.0)

    # Calendar Features
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 5)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 5)
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    
    # is_quarter_end
    df['quarter'] = (df['date'].dt.month - 1) // 3 + 1
    df['year'] = df['date'].dt.year
    df['is_quarter_end'] = 0
    for _, group in df.groupby(['year', 'quarter']):
        if len(group) >= 3:
            df.loc[group.index[-3:], 'is_quarter_end'] = 1
        else:
            df.loc[group.index, 'is_quarter_end'] = 1
            
    # days_before_tet
    if is_vn:
        df['days_before_tet'] = calculate_days_before_tet(df['date'])
    else:
        df['days_before_tet'] = 30.0

    # Áp dụng Kalman Filter để làm mịn giá đóng cửa
    try:
        from src.features import kalman_filter
    except ImportError:
        from features import kalman_filter
    df['close_smoothed'] = kalman_filter(df['close'])

    # Tích hợp đặc trưng phân tích cảm xúc tin tức tài chính
    try:
        try:
            from src.news_sentiment import get_news_sentiment_features
        except ImportError:
            from news_sentiment import get_news_sentiment_features
        df_sent = get_news_sentiment_features(ticker, df['date'].dt.strftime('%Y-%m-%d').tolist(), engine=sentiment_engine)
        df_sent['date'] = pd.to_datetime(df_sent['date'])
        df = pd.merge(df, df_sent, on='date', how='left')
    except Exception as e:
        print(f"  [WARNING] Không thể tích hợp tin tức: {e}")
        df['sentiment_score'] = 0.0
        df['news_volume'] = 0.0

    df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
    df['news_volume'] = df['news_volume'].fillna(0.0)

    # Biến mục tiêu: tỷ suất lợi nhuận mở cửa ngày mai so với đóng cửa hôm nay
    df['target_return']   = (df['open'].shift(-1) - df['close']) / df['close']
    df['target_spread']   = (df['high'].shift(-1) - df['low'].shift(-1)) / df['close']

    # Chạy DataTransformer để tính toán 34 đặc trưng nâng cao một cách nhất quán
    try:
        from src.features import DataTransformer
    except ImportError:
        from features import DataTransformer
    
    transformer = DataTransformer()
    df_feats = transformer.transform_df(df)
    
    # Merge các đặc trưng tính toán được vào DataFrame chính
    for col in df_feats.columns:
        df[col] = df_feats[col]

    df_cleaned = df.dropna().reset_index(drop=True)

    os.makedirs(data_dir, exist_ok=True)
    cache_output = os.path.join(data_dir, f"{ticker}_processed.csv")
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
            cols_show = ['date', 'close', 'rsi_14', 'macd_ratio', 'target_return']
            print("\n--- 3 phien dau ---")
            print(test_df[cols_show].head(3).to_string(index=False))
            print("\n--- 3 phien cuoi ---")
            print(test_df[cols_show].tail(3).to_string(index=False))
    except Exception as e:
        print(f"Loi kiem thu: {e}")