import os
import re
import sys
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy.orm import Session

# Add current workspace to path
# __file__ is src/web_runner/run_web.py -> up 2 levels is project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.web.backend.db import init_db, SessionLocal, Stock, StockPrice, PredictionRecord, NewsSentiment
from src.data_loader import get_realtime_usd_vnd_rate

def parse_predictions_history(db: Session):
    print("⏳ Đang phân tích file log 'logs/predict_predictions_history.txt' để khôi phục lịch sử dự báo...")
    log_path = os.path.join(ROOT_DIR, "logs", "predict_predictions_history.txt")
    if not os.path.exists(log_path):
        print("⚠️ Không tìm thấy file predict_predictions_history.txt, bỏ qua bước import lịch sử dự đoán.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"=== (?:BẢN GHI )?DỰ BÁO", content)
    imported_count = 0

    for block in blocks:
        if not block.strip():
            continue
        try:
            # Parse prediction timestamp from block header
            header_match = re.search(r"\((\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\)", block)
            if not header_match:
                continue
            pred_date = header_match.group(1)

            # Ticker
            ticker_match = re.search(r"Mã chứng khoán:\s*([A-Za-z0-9\.\-]+)", block)
            if not ticker_match:
                continue
            ticker = ticker_match.group(1).strip()

            stock = db.query(Stock).filter(Stock.ticker == ticker).first()
            if not stock:
                continue

            # Risk level
            risk_match = re.search(r"Rủi ro biến động:\s*([^\n\(\)]+)", block)
            risk_level = risk_match.group(1).strip() if risk_match else "Trung bình"

            # USD/VND rate (if exists)
            rate_match = re.search(r"Tỷ giá USD/VND quy đổi:\s*1 USD =\s*([\d\.,]+)\s*VNĐ", block)
            usd_rate = 26294.0
            if rate_match:
                usd_rate = float(rate_match.group(1).replace(".", "").replace(",", "."))

            # XGBoost Predict & Interval
            xgb_match = re.search(r"Dự báo XGBoost:\s*([\d\.,]+)\s*VNĐ.*Khoảng an toàn:\s*\[([\d\.,]+)\s*-\s*([\d\.,]+)\]\s*VNĐ", block)
            if not xgb_match:
                # Fallback for old log format
                xgb_match = re.search(r"Dự báo XGBoost:\s*([\d\.,]+)\s*VNĐ.*Khoảng an toàn:\s*\[([\d\.,]+)\s*-\s*([\d\.,]+)\]", block)
            
            if xgb_match:
                xgb_val = float(xgb_match.group(1).replace(".", "").replace(",", "."))
                xgb_lower = float(xgb_match.group(2).replace(".", "").replace(",", "."))
                xgb_upper = float(xgb_match.group(3).replace(".", "").replace(",", "."))
            else:
                continue

            # Transformer Predict & Interval
            trans_match = re.search(r"Dự báo Transformer:\s*([\d\.,]+)\s*VNĐ.*Khoảng an toàn:\s*\[([\d\.,]+)\s*-\s*([\d\.,]+)\]\s*VNĐ", block)
            if not trans_match:
                trans_match = re.search(r"Dự báo Transformer:\s*([\d\.,]+)\s*VNĐ.*Khoảng an toàn:\s*\[([\d\.,]+)\s*-\s*([\d\.,]+)\]", block)
                
            if trans_match:
                trans_val = float(trans_match.group(1).replace(".", "").replace(",", "."))
                trans_lower = float(trans_match.group(2).replace(".", "").replace(",", "."))
                trans_upper = float(trans_match.group(3).replace(".", "").replace(",", "."))
            else:
                continue

            if "VNM" not in ticker.upper():
                xgb_val = xgb_val / usd_rate
                xgb_lower = xgb_lower / usd_rate
                xgb_upper = xgb_upper / usd_rate
                trans_val = trans_val / usd_rate
                trans_lower = trans_lower / usd_rate
                trans_upper = trans_upper / usd_rate

            # Calculate target date
            p_dt = datetime.datetime.strptime(pred_date, "%Y-%m-%d")
            target_date = (p_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

            # Check duplicate
            existing = db.query(PredictionRecord).filter(
                PredictionRecord.stock_id == stock.id, 
                PredictionRecord.prediction_date == pred_date
            ).first()

            if not existing:
                db_pred = PredictionRecord(
                    stock_id=stock.id,
                    prediction_date=pred_date,
                    target_date=target_date,
                    xgb_predicted_price=xgb_val,
                    xgb_lower=xgb_lower,
                    xgb_upper=xgb_upper,
                    trans_predicted_price=trans_val,
                    trans_lower=trans_lower,
                    trans_upper=trans_upper,
                    risk_level=risk_level,
                    usd_vnd_rate=usd_rate
                )
                db.add(db_pred)
                imported_count += 1
        except Exception as e:
            print(f"⚠️ Lỗi khi parse block dự báo: {e}")
            continue

    db.commit()
    print(f"✅ Đã khôi phục thành công {imported_count} bản ghi dự báo lịch sử vào DB.")

def import_historical_prices(db: Session):
    print("⏳ Đang tải và import giá lịch sử của watchlist (VNM.VN, GOOGL, META) vào DB...")
    tickers = {
        "VNM.VN": {"name": "Vinamilk", "currency": "VND"},
        "GOOGL": {"name": "Alphabet Inc.", "currency": "USD"},
        "META": {"name": "Meta Platforms", "currency": "USD"}
    }
    
    usd_rate = get_realtime_usd_vnd_rate()

    for ticker, info in tickers.items():
        # Get or create Stock metadata
        stock = db.query(Stock).filter(Stock.ticker == ticker).first()
        if not stock:
            stock = Stock(ticker=ticker, name=info["name"], currency=info["currency"])
            db.add(stock)
            db.commit()
            db.refresh(stock)

        # Check existing prices count
        existing_count = db.query(StockPrice).filter(StockPrice.stock_id == stock.id).count()
        if existing_count > 50:
            print(f"ℹ️ Mã {ticker} đã có sẵn {existing_count} phiên giá trong DB. Bỏ qua bước import.")
            continue

        print(f"📥 Đang tải giá lịch sử cho {ticker} từ Yahoo Finance...")
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty:
                print(f"⚠️ Không tải được giá cho {ticker}")
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [col.lower() for col in df.columns]
            
            inserted = 0
            for idx, row in df.iterrows():
                date_str = idx.strftime("%Y-%m-%d")
                
                # Scale VNM if needed (DNSE < 1000 format)
                mult = 1.0
                if "VNM" in ticker.upper():
                    if row["close"] < 1000:
                        mult = 1000.0
                        
                db_price = StockPrice(
                    stock_id=stock.id,
                    date=date_str,
                    open=float(row["open"] * mult),
                    high=float(row["high"] * mult),
                    low=float(row["low"] * mult),
                    close=float(row["close"] * mult),
                    volume=float(row["volume"]) if "volume" in df.columns else 0.0
                )
                db.add(db_price)
                inserted += 1
                
            db.commit()
            print(f"✅ Đã lưu {inserted} ngày giao dịch cho {ticker} vào DB.")
        except Exception as e:
            print(f"❌ Lỗi khi tải giá lịch sử cho {ticker}: {e}")
            db.rollback()

def main():
    # Configure UTF-8 for console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("🚀 [HỆ THỐNG] Khởi tạo Cơ sở dữ liệu SQLite...")
    db_path = os.path.join(ROOT_DIR, "data", "processed", "stock_predictions.db")
    if os.path.exists(db_path):
        print("⏳ Phát hiện database cũ, tiến hành xóa để cập nhật schema mới...")
        try:
            os.remove(db_path)
            print("✅ Đã xóa database cũ thành công.")
        except Exception as e:
            print(f"⚠️ Không thể xóa database cũ: {e}")
            
    init_db()
    
    db = SessionLocal()
    try:
        # Import metadata and prices
        import_historical_prices(db)
        
        # Import log history
        parse_predictions_history(db)
        
    finally:
        db.close()

    print("🚀 [HỆ THỐNG] Khởi động Uvicorn Web Server...")
    import uvicorn
    # Run uvicorn server with app from src.web.backend.api:app relative to ROOT_DIR
    os.chdir(ROOT_DIR)
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("src.web.backend.api:app", host=host, port=8000, reload=True)



if __name__ == "__main__":
    main()
