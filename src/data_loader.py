import os
import sys
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np

# Configure UTF-8 for console output to avoid Windows charmap encoding issues
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
try:
    from src.dividend_fetcher import fetch_dividends
except ImportError:
    from dividend_fetcher import fetch_dividends
# data_loader.py — ĐÃ SỬA CÁC LỖI:
# 1. Bỏ DataTransformer() nội bộ cuối hàm fetch_and_prepare_data()
#    → hàm giờ chỉ trả df với raw price + macro, KHÔNG tự tính 34/42 features
#    → tránh dẫm lên DataTransformer của caller và tránh fit scaler sai
# 2. Kalman filter chỉ chạy 1 lần — có check "if 'close_smoothed' not in"
# 3. Regime Filter: sp500_above_ma200 (US) + vnm_etf_above_ma200 (VN)
# 4. FIX: tránh gọi lại API trùng lặp cho ticker "VNM" (VanEck Vietnam ETF)
#    — tái sử dụng market_return đã tải sẵn cho is_vn=True
# 5. FIX: dropna() chỉ áp dụng trên feature_cols THỰC SỰ dùng để train
#    (từ DataTransformer.feature_cols), tránh mất data do NaN ở cột phụ
#    (quarter, year, close_smoothed...) không nằm trong input model
# 6. FIX: xóa cột tạm 'quarter', 'year' trước khi trả về — tránh leak
#    nhầm vào features nếu code khác lỡ dùng df.columns thay vì feature_cols


def format_vn(value):
    if value is None:
        return ""
    s = f"{value:,.2f}"
    temp = s.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
    return temp


def calculate_days_before_tet(dates_series):
    tet_dates = pd.to_datetime([
        "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31",
        "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
        "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
        "2025-01-29", "2026-02-17",
    ])
    days_before = []
    dates_list = pd.to_datetime(dates_series).tolist()
    tet_last_trading_days = []
    for tet in tet_dates:
        before_tet = [d for d in dates_list if d < tet]
        tet_last_trading_days.append(max(before_tet) if before_tet else None)
    for d in dates_list:
        next_tets = [t for t in tet_last_trading_days if t is not None and t >= d]
        if next_tets:
            next_tet_last_trade = min(next_tets)
            idx_d   = dates_list.index(d)
            idx_tet = dates_list.index(next_tet_last_trade)
            days_before.append(min(idx_tet - idx_d, 30))
        else:
            days_before.append(30)
    return days_before


def get_vcb_usd_rates():
    """
    Lấy chi tiết tỷ giá USD/VND từ Vietcombank Portal (Mua tiền mặt, Mua chuyển khoản, Bán).
    Trả về dict hoặc None nếu lỗi.
    """
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        url = "https://portal.vietcombank.com.vn/Usercontrols/TVPortal.TyGia/pXML.aspx?b=10"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            for exrate in root.findall('Exrate'):
                code = exrate.get('CurrencyCode')
                if code == 'USD':
                    buy_cash = float(exrate.get('Buy', '0').replace(',', ''))
                    buy_transfer = float(exrate.get('Transfer', '0').replace(',', ''))
                    sell = float(exrate.get('Sell', '0').replace(',', ''))
                    return {
                        'buy_cash': buy_cash,
                        'buy_transfer': buy_transfer,
                        'sell': sell
                    }
    except Exception:
        pass
    return None


def get_realtime_usd_vnd_rate():
    """
    Lấy tỷ giá USD/VND realtime.
    Ưu tiên 1: Tỷ giá Mua chuyển khoản của Vietcombank (sát thực tế nhất với đầu tư tài chính).
    Ưu tiên 2: Yahoo Finance (USDVND=X).
    Ưu tiên 3: Open ER API.
    """
    # 1. Thử lấy từ Vietcombank (lấy tỷ giá Mua Chuyển Khoản)
    vcb_rates = get_vcb_usd_rates()
    if vcb_rates and 15000.0 <= vcb_rates['buy_transfer'] <= 28000.0:
        return vcb_rates['buy_transfer']

    # 2. Fallback: Yahoo Finance
    try:
        ticker = yf.Ticker("USDVND=X")
        df = ticker.history(period="5d")
        if not df.empty:
            rate = float(df['Close'].dropna().iloc[-1])
            if rate < 1000.0:
                rate *= 1000.0
            if 15000.0 <= rate <= 28000.0:
                return rate
    except Exception:
        pass

    # 3. Fallback: Open ER API
    try:
        import urllib.request, json
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            rate = float(data['rates']['VND'])
            if 15000.0 <= rate <= 28000.0:
                return rate
    except Exception:
        pass

    return 25400.0


def _clean_usdvnd(df_rate, fallback_rate):
    """Chuẩn hóa và lọc tỷ giá USD/VND — dùng chung cho mọi nơi."""
    df_rate['rate_close'] = df_rate['rate_close'].apply(
        lambda x: x * 1000.0 if x < 1000.0 else x
    )
    df_rate.loc[
        (df_rate['rate_close'] < 15000.0) | (df_rate['rate_close'] > 28000.0),
        'rate_close'
    ] = np.nan
    df_rate['rate_close'] = df_rate['rate_close'].ffill().bfill().fillna(fallback_rate)
    return df_rate


def calculate_dividend_features(df, df_divs):
    # Khởi tạo mặc định
    df['dividend_flag'] = 0.0
    df['days_to_dividend'] = 60.0
    df['days_after_dividend'] = 60.0

    if df_divs is None or df_divs.empty:
        return df

    div_dates = pd.to_datetime(df_divs['date']).dt.normalize().tolist()
    trading_dates = pd.to_datetime(df['date']).dt.normalize().tolist()

    div_trading_indices = []
    for div_date in div_dates:
        future_trades = [i for i, d in enumerate(trading_dates) if d >= div_date]
        if future_trades:
            idx = future_trades[0]
            div_trading_indices.append(idx)
            if abs((trading_dates[idx] - div_date).days) <= 3:
                df.loc[idx, 'dividend_flag'] = 1.0

    if not div_trading_indices:
        return df

    div_trading_indices = sorted(list(set(div_trading_indices)))

    days_to = []
    days_after = []

    n_days = len(trading_dates)
    for i in range(n_days):
        next_div_indices = [j for j in div_trading_indices if j >= i]
        if next_div_indices:
            days_to.append(float(min(next_div_indices[0] - i, 60)))
        else:
            days_to.append(60.0)

        prev_div_indices = [j for j in div_trading_indices if j <= i]
        if prev_div_indices:
            days_after.append(float(min(i - prev_div_indices[-1], 60)))
        else:
            days_after.append(60.0)

    df['days_to_dividend'] = days_to
    df['days_after_dividend'] = days_after
    return df


def fetch_and_prepare_data(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str = "2026-05-20",
    sentiment_engine: str = 'vader',
    is_training: bool = True,
):
    """
    Tải và chuẩn bị dữ liệu giá + macro features.

    LƯU Ý QUAN TRỌNG — ĐÃ SỬA:
    Hàm này KHÔNG còn tạo DataTransformer() nội bộ và KHÔNG tự tính
    34/42 stationary features. Caller (run_training_transformer.py, run_pipeline.py...)
    phải gọi:
        transformer = DataTransformer()
        X_scaled, y_scaled, _ = transformer.fit_transform_train_only(df)
    Điều này tránh:
    - DataTransformer nội bộ dẫm lên instance của caller
    - Scaler của caller bị fit trên features đã transform (double transform)
    """
    project_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir      = os.path.join(project_root, "data")
    base_ticker   = ticker.split('.')[0]
    school_data_path = os.path.join(data_dir, "raw", f"{base_ticker}_prices.csv")
    price_cols    = ['open', 'high', 'low', 'close']
    fallback_rate = get_realtime_usd_vnd_rate()

    # ── Tỷ giá USD/VND ────────────────────────────────────────────────
    df_usd_vnd = None
    try:
        print(f"Đang tải tỷ giá USD/VND (USDVND=X) từ {start_date} đến {end_date}...")
        raw_rate = yf.download("USDVND=X", start=start_date, end=end_date, progress=False, timeout=10)
        if not raw_rate.empty:
            df_usd_vnd = raw_rate.reset_index()
            if isinstance(df_usd_vnd.columns, pd.MultiIndex):
                df_usd_vnd.columns = [str(c[0]).lower() for c in df_usd_vnd.columns]
            else:
                df_usd_vnd.columns = [str(c).lower() for c in df_usd_vnd.columns]
            df_usd_vnd['date'] = pd.to_datetime(df_usd_vnd['date'])
            df_usd_vnd = (df_usd_vnd[['date', 'close']]
                          .rename(columns={'close': 'rate_close'})
                          .sort_values('date').reset_index(drop=True))
            df_usd_vnd = _clean_usdvnd(df_usd_vnd, fallback_rate)
    except Exception as e:
        print(f"  [WARNING] Không tải được USDVND=X: {e}")

    # ── Yahoo Finance ──────────────────────────────────────────────────
    df_yf = None
    try:
        print(f"Đang tải Yahoo Finance: {ticker}...")
        raw_data = yf.download(ticker, start=start_date, end=end_date, progress=True, timeout=10)
        if not raw_data.empty:
            df_yf = raw_data.reset_index()
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = [str(c[0]).lower() for c in df_yf.columns]
            else:
                df_yf.columns = [str(c).lower() for c in df_yf.columns]
            df_yf['date'] = pd.to_datetime(df_yf['date'])
            df_yf = df_yf.sort_values('date').reset_index(drop=True)
            if "VNM" not in ticker.upper():
                pass
            print(f"  Yahoo Finance: {len(df_yf)} phiên")
    except Exception as e:
        print(f"Không kết nối được Yahoo Finance: {e}")

    # ── Dữ liệu trường + DNSE ─────────────────────────────────────────
    if not os.path.exists(school_data_path):
        if df_yf is None or df_yf.empty:
            raise FileNotFoundError(f"Không có {school_data_path} và không kết nối Yahoo Finance!")
        df = df_yf.copy()
    else:
        print(f"Đang nạp dữ liệu trường: {school_data_path}")
        df_school = pd.read_csv(school_data_path)
        df_school.columns = [str(c).lower() for c in df_school.columns]
        if 'time' in df_school.columns:
            df_school = df_school.rename(columns={'time': 'date'})
        df_school['date'] = pd.to_datetime(df_school['date'])
        df_school = df_school.sort_values('date').reset_index(drop=True)

        # DNSE bổ sung cho VNM trước 2019
        df_dnse = pd.DataFrame()
        if "VNM" in ticker.upper():
            try:
                import urllib.request, json
                school_first_date = df_school['date'].min()
                start_epoch = int(pd.to_datetime(start_date).timestamp())
                end_epoch   = int(school_first_date.timestamp())
                url = (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
                       f"?from={start_epoch}&to={end_epoch}&symbol=VNM&resolution=1D")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as res:
                    dnse_raw = json.loads(res.read())
                if 't' in dnse_raw and dnse_raw['t'] is not None and len(dnse_raw['t']) > 0:
                    df_dnse = pd.DataFrame({
                        'date':   pd.to_datetime(dnse_raw['t'], unit='s'),
                        'open':   dnse_raw['o'], 'high': dnse_raw['h'],
                        'low':    dnse_raw['l'], 'close': dnse_raw['c'],
                        'volume': dnse_raw['v'],
                    })
                    df_dnse = df_dnse[df_dnse['date'] < school_first_date]
            except Exception as e:
                print(f"  Không tải được DNSE: {e}")

        # Quy đổi đơn vị
        keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if "VNM" in ticker.upper():
            for col in price_cols:
                if col in df_school.columns:
                    df_school[col] = df_school[col] * 1000
                if not df_dnse.empty and col in df_dnse.columns:
                    df_dnse[col] = df_dnse[col] * 1000
        else:
            pass

        school_last_date  = df_school['date'].max()
        school_first_date = df_school['date'].min()

        df_yf_new    = df_yf[df_yf['date'] > school_last_date].copy() if df_yf is not None else pd.DataFrame()
        df_yf_before = df_yf[df_yf['date'] < school_first_date].copy() if df_yf is not None else pd.DataFrame()

        frames = []
        if not df_dnse.empty:
            frames.append(df_dnse[[c for c in keep_cols if c in df_dnse.columns]])
        elif not df_yf_before.empty:
            frames.append(df_yf_before[[c for c in keep_cols if c in df_yf_before.columns]])
        frames.append(df_school[[c for c in keep_cols if c in df_school.columns]])
        if not df_yf_new.empty:
            frames.append(df_yf_new[[c for c in keep_cols if c in df_yf_new.columns]])

        df = pd.concat(frames, ignore_index=True).sort_values('date').reset_index(drop=True)
        print(f"  Tổng: {len(df)} phiên ({df['date'].min().date()} → {df['date'].max().date()})")

    # ── Feature Engineering cơ bản (macro, calendar) ─────────────────
    print("Tính macro features...")

    def _load_yf_series(sym, col_rename, pct_change=False):
        try:
            raw = yf.download(sym, start=start_date, end=end_date, progress=False, timeout=10)
            if raw.empty:
                return pd.DataFrame(columns=['date', col_rename])
            tmp = raw.reset_index()
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = [str(c[0]).lower() for c in tmp.columns]
            else:
                tmp.columns = [str(c).lower() for c in tmp.columns]
            tmp['date'] = pd.to_datetime(tmp['date'])
            if pct_change:
                tmp[col_rename] = tmp['close'].pct_change()
            else:
                tmp[col_rename] = tmp['close']
            return tmp[['date', col_rename]]
        except Exception as e:
            print(f"  [WARNING] Không tải được {sym}: {e}")
            return pd.DataFrame(columns=['date', col_rename])

    is_vn = ".VN" in ticker.upper()
    index_ticker = "VNM" if is_vn else "^IXIC"

    # FIX 4: Tải market_return MỘT LẦN. Với VN, ticker "VNM" (không .VN)
    # = VanEck Vietnam ETF trên NYSE — đồng thời dùng làm benchmark
    # market_return VÀ làm regime filter (tránh gọi API trùng lặp).
    # Cần raw close (không pct_change) để tính MA200, nên tải riêng cột close.
    df_market_raw = _load_yf_series(index_ticker, 'market_return', pct_change=True)
    df_vix    = _load_yf_series("^VIX",       'vix')
    df_tnx    = _load_yf_series("^TNX",       'bond_yield_10y')
    df_dxy    = _load_yf_series("DX-Y.NYB",   'dollar_index_change', pct_change=True)
    df_sp500  = _load_yf_series("^GSPC",      'sp500')
    df_nasdaq = _load_yf_series("^IXIC",      'nasdaq')

    df = pd.merge(df, df_market_raw, on='date', how='left')
    df['market_return']    = df['market_return'].fillna(0.0)
    df = pd.merge(df, df_vix, on='date', how='left')
    df['vix']              = df['vix'].ffill().bfill().fillna(20.0)
    df = pd.merge(df, df_tnx, on='date', how='left')
    df['bond_yield_10y']   = df['bond_yield_10y'].ffill().bfill().fillna(4.0)
    df = pd.merge(df, df_dxy, on='date', how='left')
    df['dollar_index_change'] = df['dollar_index_change'].fillna(0.0)

    # Thêm S&P 500 và NASDAQ cho Regime Detection (luôn tải — dùng cho
    # nasdaq_12m_return của cả US và VN, theo logic gốc).
    df = pd.merge(df, df_sp500, on='date', how='left')
    df['sp500'] = df['sp500'].ffill().bfill().fillna(4000.0)
    df = pd.merge(df, df_nasdaq, on='date', how='left')
    df['nasdaq'] = df['nasdaq'].ffill().bfill().fillna(15000.0)

    # Tính toán chỉ báo Regime cho US (SP500)
    sp500_ma200 = df['sp500'].rolling(200, min_periods=1).mean()
    df['sp500_above_ma200'] = (df['sp500'] > sp500_ma200).astype(float)
    df['nasdaq_12m_return'] = (df['nasdaq'] / df['nasdaq'].shift(252) - 1).fillna(0.0)

    # FIX 4: Chỉ báo Regime cho thị trường Việt Nam — TÁI SỬ DỤNG
    # df_market_raw (ticker "VNM" = VanEck Vietnam ETF) đã tải ở trên,
    # không gọi lại API. Cần raw close (không pct_change) cho MA200,
    # nên tải lại CHỈ close (không pct_change) qua 1 lần gọi riêng,
    # nhẹ hơn việc tải y_xs+pct_change hai lần.
    if is_vn:
        df_vnm_etf_close = _load_yf_series("VNM", 'vnm_etf', pct_change=False)
        if not df_vnm_etf_close.empty:
            df = pd.merge(df, df_vnm_etf_close, on='date', how='left')
            df['vnm_etf'] = df['vnm_etf'].ffill().bfill().fillna(df['vnm_etf'].median())
            vnm_ma200 = df['vnm_etf'].rolling(200, min_periods=1).mean()
            df['vnm_etf_above_ma200'] = (df['vnm_etf'] > vnm_ma200).astype(float)
        else:
            df['vnm_etf_above_ma200'] = 1.0  # fallback: không chặn mua
    else:
        df['vnm_etf_above_ma200'] = 1.0

    # USD/VND change
    if df_usd_vnd is not None:
        df_usd_vnd_ch = df_usd_vnd.copy()
        df_usd_vnd_ch['usdvnd_change'] = df_usd_vnd_ch['rate_close'].pct_change()
        df = pd.merge(df, df_usd_vnd_ch[['date', 'usdvnd_change']], on='date', how='left')
    else:
        df['usdvnd_change'] = 0.0
    df['usdvnd_change'] = df['usdvnd_change'].fillna(0.0)

    # Đồng bộ múi giờ — US macro shift(1) cho VN tickers (NYSE đóng cửa
    # sau HOSE), không shift cho US tickers (cùng phiên giao dịch).
    if is_vn:
        df['vix_lag1']            = df['vix'].shift(1)
        df['bond_yield_lag1']     = df['bond_yield_10y'].shift(1)
        df['usdvnd_change']       = df['usdvnd_change'].shift(1)
        df['vnindex_return_lag1'] = df['market_return'].shift(1)
        df['sp500_above_ma200']   = df['sp500_above_ma200'].shift(1)
        df['nasdaq_12m_return']   = df['nasdaq_12m_return'].shift(1)
    else:
        df['vix_lag1']            = df['vix']
        df['bond_yield_lag1']     = df['bond_yield_10y']
        df['usdvnd_change']       = df['dollar_index_change']
        df['vnindex_return_lag1'] = df['market_return']
        # sp500_above_ma200 / nasdaq_12m_return: KHÔNG shift cho US —
        # SP500 close cùng ngày T hợp lệ cho US tickers (cùng timezone).

    df['vix_lag1']        = df['vix_lag1'].ffill().bfill().fillna(20.0)
    df['bond_yield_lag1'] = df['bond_yield_lag1'].ffill().bfill().fillna(4.0)
    df['sp500_above_ma200'] = df['sp500_above_ma200'].ffill().bfill().fillna(1.0)
    df['nasdaq_12m_return'] = df['nasdaq_12m_return'].ffill().bfill().fillna(0.0)

    # Calendar features
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 5)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 5)
    df['month_sin']       = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos']       = np.cos(2 * np.pi * df['date'].dt.month / 12)

    # FIX 6: 'quarter' và 'year' chỉ dùng tạm để tính is_quarter_end,
    # sẽ bị xóa ở cuối hàm trước khi trả về (tránh leak vào features
    # nếu code khác lỡ dùng df.columns thay vì feature_cols cố định).
    df['quarter'] = (df['date'].dt.month - 1) // 3 + 1
    df['year']    = df['date'].dt.year
    df['is_quarter_end'] = 0
    for _, group in df.groupby(['year', 'quarter']):
        if len(group) >= 3:
            df.loc[group.index[-3:], 'is_quarter_end'] = 1
        else:
            df.loc[group.index, 'is_quarter_end'] = 1

    df['days_before_tet'] = calculate_days_before_tet(df['date']) if is_vn else 30.0

    # === Kalman filter với guard — chỉ tính 1 lần ===
    # Tránh double-smooth nếu df đã có close_smoothed (ví dụ load từ cache)
    if 'close_smoothed' not in df.columns:
        try:
            from src.features import kalman_filter
        except ImportError:
            from features import kalman_filter
        df['close_smoothed'] = kalman_filter(df['close'])
    else:
        print("  [INFO] close_smoothed đã có trong df, bỏ qua Kalman (tránh double-smooth).")

    # Sentiment
    try:
        try:
            from src.news_sentiment import get_news_sentiment_features
        except ImportError:
            from news_sentiment import get_news_sentiment_features
        df_sent = get_news_sentiment_features(
            ticker, df['date'].dt.strftime('%Y-%m-%d').tolist(), engine=sentiment_engine
        )
        df_sent['date'] = pd.to_datetime(df_sent['date'])
        df = pd.merge(df, df_sent, on='date', how='left')
    except Exception as e:
        print(f"  [WARNING] Không tích hợp được tin tức: {e}")
        df['sentiment_score'] = 0.0
        df['news_volume']     = 0.0

    df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
    df['news_volume']     = df['news_volume'].fillna(0.0)

    # Tính đặc trưng cổ tức
    try:
        df_divs = fetch_dividends(ticker, start_date, end_date)
        df = calculate_dividend_features(df, df_divs)
    except Exception as e:
        print(f"  [WARNING] Lỗi tính đặc trưng cổ tức: {e}")
        df['dividend_flag'] = 0.0
        df['days_to_dividend'] = 60.0
        df['days_after_dividend'] = 60.0

    # Target
    for h in [1, 2, 3]:
        df[f'target_return_{h}d'] = (df['close'].shift(-h) - df['close']) / df['close']
        df[f'target_spread_{h}d'] = (df['high'].shift(-h) - df['low'].shift(-h)) / df['close']

    # === KHÔNG còn tạo DataTransformer() để transform_df() ở đây ===
    # Trước: transformer = DataTransformer(); df_feats = transformer.transform_df(df)
    # → gây double-transform và scaler leak
    # Sau: caller tự gọi transformer.fit_transform_train_only(df)
    #
    # NHƯNG: vẫn cần biết DANH SÁCH feature_cols thực sự (42 cột) để
    # dropna() đúng phạm vi — import DataTransformer CHỈ để đọc
    # .feature_cols (không gọi transform_df/fit ở đây, không leak).
    try:
        from src.features import DataTransformer
    except ImportError:
        from features import DataTransformer

    _temp_dt = DataTransformer()
    target_cols = [c for c in df.columns if c.startswith('target_')]

    # FIX 5: dropna() chỉ trên các cột THỰC SỰ dùng làm input model
    # (42 feature_cols) + target_cols. Các cột phụ trợ tạm thời
    # (close_smoothed, sentiment_score nếu chưa có trong feature_cols,
    # vnm_etf, sp500, nasdaq...) KHÔNG bắt buộc phải non-NaN, vì chúng
    # chỉ là nguyên liệu trung gian để tính feature_cols qua transform_df(),
    # không phải input trực tiếp.
    required_cols = [c for c in _temp_dt.feature_cols if c in df.columns]
    # Nếu một feature_col chưa tồn tại trực tiếp trong df ở giai đoạn này
    # (vì nó được TÍNH trong transform_df(), ví dụ 'rsi_14', 'mfi_14'),
    # ta KHÔNG thể dropna trên nó ở đây — chỉ dropna trên các cột macro/
    # passthrough đã thực sự có mặt trong df tại bước này.
    if is_training:
        dropna_cols = list(dict.fromkeys(required_cols + target_cols))
    else:
        dropna_cols = required_cols

    # FIX 6: xóa cột tạm 'quarter', 'year' trước khi trả về
    df = df.drop(columns=['quarter', 'year'], errors='ignore')

    df_cleaned = df.dropna(subset=dropna_cols).reset_index(drop=True)

    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"{ticker}_processed.csv")
    df_cleaned.to_csv(cache_path, index=False)
    print(f"Đã lưu cache: {cache_path}")
    print(f"Sẵn sàng với {len(df_cleaned)} phiên.\n")
    return df_cleaned


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Cấu hình UTF-8 cho console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=== CHẠY TẢI DỮ LIỆU GIAO DỊCH & VĨ MÔ ===")
    tickers = ["VNM.VN", "GOOGL", "META"]
    for ticker in tickers:
        print(f"\n📥 Đang tải và tiền xử lý dữ liệu cho: {ticker}")
        try:
            df = fetch_and_prepare_data(ticker, start_date="2015-01-01", end_date="2026-05-20")
            print(f"   => Hoàn thành! Số lượng phiên giao dịch: {len(df)}")
            print(f"   => Khoảng thời gian: {df['date'].min().date()} đến {df['date'].max().date()}")
        except Exception as e:
            print(f"   => Lỗi: {e}")