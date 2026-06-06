import os
import sys
import datetime
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import joblib
import tensorflow as tf

# Add root project folder to sys.path so that src can be imported correctly
# __file__ is src/web/backend/api.py -> up 3 levels is project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)

from src.web.backend.db import get_db, Stock, StockPrice, NewsSentiment, PredictionRecord, init_db
from src.ai_models import PositionalEmbedding, TimeDecayAttention, MultiTaskModel, UncertaintyWeightsLayer
from src.data_loader import format_vn, get_realtime_usd_vnd_rate
from src.features import DataTransformer


app = FastAPI(title="Stock Prediction API", version="1.0.0")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature columns used by the models (22 stationary features)
FEATURE_COLS = DataTransformer().feature_cols

LOOKBACK_WINDOW = 45

# Root helper to check status
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "time": datetime.datetime.now().isoformat()}

# Get all stocks
@app.get("/api/stocks")
def get_stocks(db: Session = Depends(get_db)):
    stocks = db.query(Stock).all()
    return [{"id": s.id, "ticker": s.ticker, "name": s.name, "currency": s.currency} for s in stocks]

# Get historical prices for charting
@app.get("/api/prices/{ticker}")
def get_stock_prices(ticker: str, limit: int = 150, db: Session = Depends(get_db)):
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Ticker not found in database")
    
    prices = db.query(StockPrice).filter(StockPrice.stock_id == stock.id).order_by(StockPrice.date.asc()).all()
    return [
        {
            "date": p.date,
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume
        } for p in prices[-limit:]
    ]

# Get predictions history and latest prediction
@app.get("/api/predictions/{ticker}")
def get_predictions(ticker: str, limit: int = 10, db: Session = Depends(get_db)):
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Ticker not found in database")
    
    preds = db.query(PredictionRecord).filter(PredictionRecord.stock_id == stock.id).order_by(PredictionRecord.prediction_date.desc()).limit(limit).all()
    return [
        {
            "id": p.id,
            "prediction_date": p.prediction_date,
            "target_date": p.target_date,
            "xgb_predicted_price": p.xgb_predicted_price,
            "xgb_lower": p.xgb_lower,
            "xgb_upper": p.xgb_upper,
            "trans_predicted_price": p.trans_predicted_price,
            "trans_lower": p.trans_lower,
            "trans_upper": p.trans_upper,
            "risk_level": p.risk_level,
            "usd_vnd_rate": p.usd_vnd_rate,
            "actual_open_price": p.actual_open_price
        } for p in preds
    ]

# Get latest news sentiments
@app.get("/api/news/{ticker}")
def get_news_sentiments(ticker: str, limit: int = 10, db: Session = Depends(get_db)):
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Ticker not found in database")
    
    news = db.query(NewsSentiment).filter(NewsSentiment.stock_id == stock.id).order_by(NewsSentiment.published_date.desc()).limit(limit).all()
    return [
        {
            "id": n.id,
            "published_date": n.published_date,
            "title": n.title,
            "source": n.source,
            "sentiment_score": n.sentiment_score,
            "sentiment_label": n.sentiment_label
        } for n in news
    ]

# Execute prediction logic online for a ticker and save it to the DB
@app.post("/api/predict/trigger/{ticker}")
def trigger_prediction(ticker: str, db: Session = Depends(get_db)):
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Ticker not found in database")
    
    # Path validation for model assets
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
    xgb_path = os.path.join(models_dir, f"xgboost_model_{stock.ticker}.pkl")
    trans_path = os.path.join(models_dir, f"transformer_model_{stock.ticker}.keras")
    scaler_x_path = os.path.join(models_dir, f"feature_scaler_{stock.ticker}.pkl")
    scaler_y_path = os.path.join(models_dir, f"target_scaler_{stock.ticker}.pkl")
    
    if not (os.path.exists(xgb_path) and os.path.exists(trans_path) and os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path)):
        raise HTTPException(
            status_code=400, 
            detail=f"Mô hình cho mã {stock.ticker} chưa được huấn luyện. Vui lòng chạy huấn luyện trước từ console."
        )
        
    try:
        # Load scalers and models
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        xgb_model = joblib.load(xgb_path)
        transformer_model = tf.keras.models.load_model(
            trans_path, 
            custom_objects={
                'PositionalEmbedding': PositionalEmbedding,
                'TimeDecayAttention': TimeDecayAttention,
                'MultiTaskModel': MultiTaskModel,
                'UncertaintyWeightsLayer': UncertaintyWeightsLayer
            },
            safe_mode=False
        )
        
        # Tải và tiền xử lý dữ liệu bằng data loader đồng bộ
        usd_vnd_rate = get_realtime_usd_vnd_rate()
        start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
        end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        from src.data_loader import fetch_and_prepare_data
        df = fetch_and_prepare_data(stock.ticker, start_date=start_date, end_date=end_date, sentiment_engine="vader")
        
        if df.empty:
            raise HTTPException(status_code=500, detail="Không thể tải và chuẩn bị dữ liệu giao dịch mới nhất")
            
        # Tính toán chỉ báo ATR 14 để đo lường rủi ro và vẽ khoảng an toàn
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        transformer = DataTransformer(time_steps=LOOKBACK_WINDOW)
        df = df.sort_values('date').reset_index(drop=True)
        
        recent_features = df[transformer.feature_cols].tail(LOOKBACK_WINDOW)
        if len(recent_features) < LOOKBACK_WINDOW:
            raise HTTPException(status_code=500, detail=f"Không đủ dữ liệu {LOOKBACK_WINDOW} ngày để tạo đặc trưng")
            
        raw_df = df.tail(LOOKBACK_WINDOW).copy()
        raw_df.set_index('date', inplace=True)
        
        # Scale inputs
        recent_scaled = scaler_X.transform(recent_features.values)
        X_predict = recent_scaled.reshape(1, LOOKBACK_WINDOW, len(FEATURE_COLS))
        
        # Gọi mô hình với dữ liệu để khởi tạo thuộc tính input/output của đồ thị Functional
        _ = transformer_model(X_predict)
        
        # Trích xuất đặc trưng lai (32 chiều ẩn từ Transformer + 34 chiều chỉ báo ngày hiện tại) cho mô hình Hybrid XGBoost
        feature_extractor = tf.keras.models.Model(
            inputs=transformer_model.input,
            outputs=transformer_model.get_layer("latent_embedding").output
        )
        X_predict_latent = feature_extractor.predict(X_predict, verbose=0)
        X_predict_today = X_predict[0, -1, :].reshape(1, -1)
        X_predict_hybrid = np.concatenate([X_predict_latent, X_predict_today], axis=1)

        
        xgb_pred_scaled = xgb_model.predict(X_predict_hybrid).reshape(-1, 1)
        xgb_return_future = scaler_y.inverse_transform(xgb_pred_scaled)[0][0]
        
        trans_pred_output = transformer_model.predict(X_predict, verbose=0)
        trans_scaled = trans_pred_output[0] if isinstance(trans_pred_output, list) else trans_pred_output
        trans_return_future = scaler_y.inverse_transform(trans_scaled)[0][0]

        
        last_close = float(raw_df['close'].iloc[-1])
        last_date_str = raw_df.index[-1].strftime('%Y-%m-%d')
        target_date_str = (raw_df.index[-1] + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

        
        xgb_val = float(last_close * (1 + xgb_return_future))
        trans_val = float(last_close * (1 + trans_return_future))
        
        last_atr = float(raw_df['atr_14'].iloc[-1])
        risk_ratio = (last_atr / last_close) * 100
        if risk_ratio < 1.5:
            risk_level = "Thấp 🟢"
        elif risk_ratio < 3.0:
            risk_level = "Trung bình 🟡"
        else:
            risk_level = "Cao 🔴"
            
        xgb_lower = float(xgb_val - 1.5 * last_atr)
        xgb_upper = float(xgb_val + 1.5 * last_atr)
        trans_lower = float(trans_val - 1.5 * last_atr)
        trans_upper = float(trans_val + 1.5 * last_atr)

        
        # Save historical price record to database
        db_price = db.query(StockPrice).filter(StockPrice.stock_id == stock.id, StockPrice.date == last_date_str).first()
        if not db_price:
            db_price = StockPrice(
                stock_id=stock.id,
                date=last_date_str,
                open=float(raw_df['open'].iloc[-1]),
                high=float(raw_df['high'].iloc[-1]),
                low=float(raw_df['low'].iloc[-1]),
                close=last_close,
                volume=float(raw_df['volume'].iloc[-1]) if 'volume' in raw_df.columns else 0.0
            )
            db.add(db_price)
            
        # Save prediction to DB
        db_pred = db.query(PredictionRecord).filter(
            PredictionRecord.stock_id == stock.id, 
            PredictionRecord.prediction_date == last_date_str
        ).first()
        
        if db_pred:
            db_pred.target_date = target_date_str
            db_pred.xgb_predicted_price = xgb_val
            db_pred.xgb_lower = xgb_lower
            db_pred.xgb_upper = xgb_upper
            db_pred.trans_predicted_price = trans_val
            db_pred.trans_lower = trans_lower
            db_pred.trans_upper = trans_upper
            db_pred.risk_level = risk_level
            db_pred.usd_vnd_rate = usd_vnd_rate
        else:
            db_pred = PredictionRecord(
                stock_id=stock.id,
                prediction_date=last_date_str,
                target_date=target_date_str,
                xgb_predicted_price=xgb_val,
                xgb_lower=xgb_lower,
                xgb_upper=xgb_upper,
                trans_predicted_price=trans_val,
                trans_lower=trans_lower,
                trans_upper=trans_upper,
                risk_level=risk_level,
                usd_vnd_rate=usd_vnd_rate
            )
            db.add(db_pred)
            
        db.commit()
        
        return {
            "ticker": stock.ticker,
            "prediction_date": last_date_str,
            "target_date": target_date_str,
            "last_close": last_close,
            "xgb_predicted_price": xgb_val,
            "xgb_lower": xgb_lower,
            "xgb_upper": xgb_upper,
            "trans_predicted_price": trans_val,
            "trans_lower": trans_lower,
            "trans_upper": trans_upper,
            "risk_level": risk_level,
            "usd_vnd_rate": usd_vnd_rate
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi chạy dự báo: {str(e)}")

# Mount static frontend files
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

from fastapi.responses import FileResponse

@app.get("/")
def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="web")
