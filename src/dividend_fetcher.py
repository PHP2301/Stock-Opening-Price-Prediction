import sys
import yfinance as yf
import pandas as pd

# Configure UTF-8 for console output to avoid Windows charmap encoding issues
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def fetch_dividends(ticker: str, start_date: str = "2015-01-01", end_date: str = "2026-05-20") -> pd.DataFrame:
    """
    Tải thông tin cổ tức lịch sử của một mã cổ phiếu từ yfinance.
    
    Parameters:
        ticker (str): Mã cổ phiếu (ví dụ: VNM.VN, GOOGL, META)
        start_date (str): Ngày bắt đầu dạng YYYY-MM-DD
        end_date (str): Ngày kết thúc dạng YYYY-MM-DD
        
    Returns:
        pd.DataFrame: DataFrame có 2 cột ['date', 'dividend_amount']
    """
    try:
        print(f"📥 Đang tải dữ liệu cổ tức cho {ticker} từ yfinance...")
        stock = yf.Ticker(ticker)
        divs = stock.dividends
        
        if divs.empty:
            print(f"   ℹ️ Không có lịch sử cổ tức cho {ticker}")
            return pd.DataFrame(columns=['date', 'dividend_amount'])
        
        df_divs = divs.reset_index()
        # Chuẩn hóa tên cột
        df_divs.columns = ['date', 'dividend_amount']
        
        # Chuyển đổi date thành timezone-naive date
        df_divs['date'] = pd.to_datetime(df_divs['date']).dt.tz_localize(None).dt.normalize()
        
        # Lọc theo khoảng thời gian yêu cầu
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df_divs = df_divs[(df_divs['date'] >= start_dt) & (df_divs['date'] <= end_dt)]
        
        df_divs = df_divs.sort_values('date').reset_index(drop=True)
        print(f"   => Đã tải {len(df_divs)} đợt chia cổ tức cho {ticker}")
        return df_divs
        
    except Exception as e:
        print(f"   ⚠️ Lỗi khi tải cổ tức của {ticker}: {e}")
        return pd.DataFrame(columns=['date', 'dividend_amount'])

if __name__ == "__main__":
    # Test chạy thử
    df = fetch_dividends("VNM.VN")
    if not df.empty:
        print(df.head())
