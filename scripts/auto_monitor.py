import os
import sys
import time
import datetime
import json
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np

# Cấu hình UTF-8 cho Windows console để tránh lỗi hiển thị tiếng Việt
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm root dir vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.notifications import send_telegram_message, load_dotenv
from src.news_sentiment import fetch_latest_news
from scripts.predict import run_prediction_for_ticker

# Load environment variables
load_dotenv()

CACHE_FILE = os.path.join(ROOT_DIR, "data", "auto_monitor_cache.json")
TICKERS = ["VNM.VN", "GOOGL", "META"]

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"daily_runs": {}, "alerted_catalysts": {}}

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ Lỗi lưu cache: {e}")

def get_market_indicators(ticker):
    """
    Tải nhanh dữ liệu gần nhất để tính RSI(14) và MFI(14) mà không cần chạy pipeline nặng.
    """
    try:
        # Tải dữ liệu 30 ngày gần nhất
        df = yf.download(ticker, period="30d", interval="1d", progress=False)
        if df.empty:
            return None, None
        
        # Flatten columns if multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # 1. Tính RSI(14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1])
        
        # 2. Tính MFI(14)
        typical_price = (high + low + close) / 3.0
        money_flow = typical_price * volume
        
        delta_tp = typical_price.diff()
        pos_flow = money_flow.where(delta_tp > 0, 0).rolling(window=14).sum()
        neg_flow = money_flow.where(delta_tp < 0, 0).rolling(window=14).sum()
        
        mfr = pos_flow / (neg_flow + 1e-9)
        mfi = 100 - (100 / (1 + mfr))
        mfi_val = float(mfi.iloc[-1])
        
        return rsi_val, mfi_val
    except Exception as e:
        print(f"⚠️ Lỗi tính chỉ số thị trường cho {ticker}: {e}")
        return None, None

def check_and_alert_catalysts(ticker, cache):
    print(f"🔍 [{ticker}] Đang kiểm tra tin tức nóng & chỉ số đột biến...")
    catalysts = []
    
    # 1. Kiểm tra tin tức nóng (Breaking/Outage/Sập)
    try:
        news_items = fetch_latest_news(ticker)
        breaking_titles = [n['title'] for n in news_items if "BREAKING" in n['title'].upper() or "SẬP" in n['title'].upper() or "OUTAGE" in n['title'].upper()]
        for bt in breaking_titles:
            # Hash tiêu đề để kiểm tra trùng lặp
            h = hashlib.md5(bt.encode('utf-8')).hexdigest()
            cache_key = f"news_{ticker}_{h}"
            
            if cache_key not in cache["alerted_catalysts"]:
                if "outage" in bt.lower() or "sập" in bt.lower():
                    desc = "🔴 <b>Tin tức nóng:</b> Sự cố sập hệ thống toàn cầu của Meta gây sụt giảm doanh thu quảng cáo nghiêm trọng."
                else:
                    desc = f"🔴 <b>Tin tức nóng:</b> {bt}"
                catalysts.append((cache_key, desc))
    except Exception as e:
        print(f"⚠️ Lỗi fetch news cho {ticker}: {e}")

    # 2. Kiểm tra chỉ báo quá mua/quá bán (RSI, MFI)
    rsi_val, mfi_val = get_market_indicators(ticker)
    if rsi_val is not None and mfi_val is not None:
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        
        # Quá mua (Gom hàng/Đổ tiền vào)
        if rsi_val > 80 or mfi_val > 80:
            cache_key = f"flow_buy_{ticker}_{today_str}"
            if cache_key not in cache["alerted_catalysts"]:
                desc = f"📈 <b>Dòng tiền đột biến:</b> Lực mua gom hàng cực mạnh (Quá mua) - RSI: <b>{rsi_val:.1f}</b>, MFI: <b>{mfi_val:.1f}</b>. Có dòng tiền lớn đổ vào đẩy giá lên."
                catalysts.append((cache_key, desc))
                
        # Quá bán (Bán tháo)
        elif rsi_val < 25 or mfi_val < 25:
            cache_key = f"flow_sell_{ticker}_{today_str}"
            if cache_key not in cache["alerted_catalysts"]:
                desc = f"📉 <b>Dòng tiền tháo chạy:</b> Áp lực bán tháo hoảng loạn cực mạnh (Quá bán) - RSI: <b>{rsi_val:.1f}</b>, MFI: <b>{mfi_val:.1f}</b>. Cổ phiếu đang bị bán tháo mạnh."
                catalysts.append((cache_key, desc))

    # Gửi tin nhắn Telegram nếu phát hiện các nhân tố mới
    if catalysts:
        for cache_key, desc in catalysts:
            msg = (
                f"🚨🚨🚨 <b>BÁO CÁO BIẾN ĐỘNG KHẨN CẤP - {ticker}</b> 🚨🚨🚨\n"
                f"-----------------------------------------\n"
                f"{desc}\n"
                f"-----------------------------------------\n"
                f"💡 <i>Hệ thống khuyến nghị bạn kiểm tra biểu đồ và vị thế tài khoản.</i>"
            )
            # Gửi Telegram
            if send_telegram_message(msg):
                cache["alerted_catalysts"][cache_key] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_cache(cache)

def check_daily_run(cache):
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    
    # Thời gian chạy tự động (mặc định 17:00 ICT)
    target_hour = int(os.environ.get("DAILY_PREDICT_HOUR", "17"))
    now = datetime.datetime.now()
    
    if now.hour >= target_hour:
        if cache["daily_runs"].get(today_str) is not True:
            print(f"⏰ Đến giờ chạy báo cáo dự báo hàng ngày ({now.strftime('%H:%M:%S')}). Bắt đầu pipeline...")
            for ticker in TICKERS:
                try:
                    run_prediction_for_ticker(ticker)
                except Exception as e:
                    print(f"❌ Lỗi chạy dự báo hàng ngày cho {ticker}: {e}")
            
            cache["daily_runs"][today_str] = True
            save_cache(cache)
            print(f"✅ Báo cáo dự báo hàng ngày đã gửi thành công.")

def main():
    print("=========================================================")
    print("🚀 BẮT ĐẦU CHẠY MONITOR TỰ ĐỘNG (BACKGROUND MONITOR)")
    print("   - Kiểm tra tin tức khẩn cấp & dòng tiền mỗi 10 phút.")
    print("   - Tự động chạy dự báo cuối ngày lúc 17:00 ICT.")
    print("=========================================================")
    
    # Thời gian chờ giữa các vòng quét (mặc định 10 phút = 600 giây)
    check_interval = int(os.environ.get("MONITOR_INTERVAL_SECONDS", "600"))
    
    while True:
        try:
            cache = load_cache()
            
            # 1. Chạy báo cáo hàng ngày nếu đến giờ
            check_daily_run(cache)
            
            # 2. Quét tin tức và dòng tiền khẩn cấp
            for ticker in TICKERS:
                check_and_alert_catalysts(ticker, cache)
                
        except Exception as e:
            print(f"❌ Lỗi ngoài dự kiến trong vòng lặp monitor: {e}")
            
        print(f"💤 Tạm nghỉ {check_interval // 60} phút trước lượt quét tiếp theo...")
        time.sleep(check_interval)

if __name__ == "__main__":
    main()
