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
from src.ai_models import PositionalEmbedding
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
        transformer_model = tf.keras.models.load_model(trans_path, custom_objects={'PositionalEmbedding': PositionalEmbedding})
        
        # Download recent data for indicators
        raw_df = yf.download(stock.ticker, period="150d", progress=False)
        if raw_df.empty:
            raise HTTPException(status_code=500, detail="Không thể tải dữ liệu giao dịch mới nhất từ Yahoo Finance")
            
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.droplevel(1)
        raw_df.columns = [col.lower() for col in raw_df.columns]
        
        # Real-time exchange rate
        usd_vnd_rate = get_realtime_usd_vnd_rate()
        df_rate_hist = pd.DataFrame()
        try:
            df_rate_hist = yf.download("USDVND=X", period="150d", progress=False)
            if not df_rate_hist.empty:
                if isinstance(df_rate_hist.columns, pd.MultiIndex):
                    df_rate_hist.columns = [col[0].lower() for col in df_rate_hist.columns]
                else:
                    df_rate_hist.columns = [col.lower() for col in df_rate_hist.columns]
                df_rate_hist = df_rate_hist[['close']].rename(columns={'close': 'rate_close'})
                df_rate_hist['rate_close'] = df_rate_hist['rate_close'].apply(lambda x: x * 1000.0 if x < 1000.0 else x)
                df_rate_hist.loc[(df_rate_hist['rate_close'] < 15000.0) | (df_rate_hist['rate_close'] > 28000.0), 'rate_close'] = np.nan
                df_rate_hist['rate_close'] = df_rate_hist['rate_close'].ffill().bfill().fillna(usd_vnd_rate)
                usd_vnd_rate = float(df_rate_hist['rate_close'].iloc[-1])
        except Exception as ex:
            print(f"Error fetching USDVND: {ex}")
            
        # Convert prices to VND if US stock
        if "VNM" not in stock.ticker.upper():
            if not df_rate_hist.empty:
                raw_df = raw_df.merge(df_rate_hist, left_index=True, right_index=True, how='left')
                raw_df['rate_close'] = raw_df['rate_close'].ffill().bfill().fillna(usd_vnd_rate)
                for col in ['open', 'high', 'low', 'close']:
                    raw_df[col] = raw_df[col] * raw_df['rate_close']
            else:
                for col in ['open', 'high', 'low', 'close']:
                    raw_df[col] = raw_df[col] * usd_vnd_rate
        else:
            # For VNM, convert values < 1000 to full VND
            for col in ['open', 'high', 'low', 'close']:
                raw_df[col] = raw_df[col].apply(lambda x: x * 1000.0 if x < 1000.0 else x)
                
        raw_df = raw_df.reset_index()
        raw_df.rename(columns={'Date': 'date'}, inplace=True)
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        
        # Load market index and VIX
        market_ticker = "VNM" if "VNM" in stock.ticker.upper() else "^GSPC"
        try:
            m_df = yf.download(market_ticker, period="150d", progress=False)
            if isinstance(m_df.columns, pd.MultiIndex):
                m_df.columns = m_df.columns.droplevel(1)
            m_df.columns = [col.lower() for col in m_df.columns]
            m_df = m_df[['close']].rename(columns={'close': 'market_close'})
            m_df['market_return'] = m_df['market_close'].pct_change()
            raw_df = pd.merge(raw_df, m_df[['market_return']], left_on='date', right_index=True, how='left')
        except Exception:
            raw_df['market_return'] = 0.0
            
        try:
            vix_df = yf.download("^VIX", period="150d", progress=False)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.droplevel(1)
            vix_df.columns = [col.lower() for col in vix_df.columns]
            vix_df = vix_df[['close']].rename(columns={'close': 'vix_close'})
            raw_df = pd.merge(raw_df, vix_df, left_on='date', right_index=True, how='left')
        except Exception:
            raw_df['vix_close'] = 20.0

        try:
            tnx_df = yf.download("^TNX", period="150d", progress=False)
            if isinstance(tnx_df.columns, pd.MultiIndex):
                tnx_df.columns = tnx_df.columns.droplevel(1)
            tnx_df.columns = [col.lower() for col in tnx_df.columns]
            tnx_df = tnx_df[['close']].rename(columns={'close': 'bond_yield_10y'})
            raw_df = pd.merge(raw_df, tnx_df, left_on='date', right_index=True, how='left')
        except Exception:
            raw_df['bond_yield_10y'] = 4.0

        try:
            dxy_df = yf.download("DX-Y.NYB", period="150d", progress=False)
            if isinstance(dxy_df.columns, pd.MultiIndex):
                dxy_df.columns = dxy_df.columns.droplevel(1)
            dxy_df.columns = [col.lower() for col in dxy_df.columns]
            dxy_df['dollar_index_change'] = dxy_df['close'].pct_change()
            raw_df = pd.merge(raw_df, dxy_df[['dollar_index_change']], left_on='date', right_index=True, how='left')
        except Exception:
            raw_df['dollar_index_change'] = 0.0
            
        # Merge news sentiment features
        try:
            from src.news_sentiment import get_news_sentiment_features
            df_sent_pred = get_news_sentiment_features(stock.ticker, raw_df['date'].dt.strftime('%Y-%m-%d').tolist(), engine='vader')
            df_sent_pred['date'] = pd.to_datetime(df_sent_pred['date'])
            raw_df = pd.merge(raw_df, df_sent_pred, on='date', how='left')
        except Exception:
            raw_df['sentiment_score'] = 0.0
            raw_df['news_volume'] = 0.0
            
        raw_df['sentiment_score'] = raw_df['sentiment_score'].fillna(0.0)
        raw_df['news_volume'] = raw_df['news_volume'].fillna(0.0)
        raw_df['market_return'] = raw_df['market_return'].fillna(0.0)
        raw_df['vix_close'] = raw_df['vix_close'].fillna(20.0)
        raw_df['bond_yield_10y'] = raw_df['bond_yield_10y'].fillna(4.0)
        raw_df['dollar_index_change'] = raw_df['dollar_index_change'].fillna(0.0)
        
        # Apply Kalman Filter to smooth price
        try:
            from src.features import kalman_filter
        except ImportError:
            from features import kalman_filter
        raw_df['close_smoothed'] = kalman_filter(raw_df['close'])
        
        # Calculate indicators on close_smoothed
        raw_df.set_index('date', inplace=True)
        raw_df['rsi_14'] = ta.rsi(raw_df['close_smoothed'], length=14)
        raw_df['rsi_lag1'] = raw_df['rsi_14'].shift(1)
        macd_df = ta.macd(raw_df['close_smoothed'], fast=12, slow=26, signal=9)
        raw_df = pd.concat([raw_df, macd_df], axis=1)
        
        macd_cols_to_drop = [col for col in macd_df.columns if 'MACDh' in col or 'MACDs' in col]
        raw_df = raw_df.drop(columns=macd_cols_to_drop)
        macd_col_name = [col for col in raw_df.columns if 'MACD_12_26_9' in col or 'macd' in col.lower()][0]
        raw_df.rename(columns={macd_col_name: 'macd_12_26_9'}, inplace=True)
        
        raw_df['volatility_20'] = raw_df['close_smoothed'].pct_change().rolling(window=20).std()
        raw_df['close_lag1'] = raw_df['close_smoothed'].shift(1)
        raw_df['close_lag2'] = raw_df['close_smoothed'].shift(2)
        raw_df['close_lag3'] = raw_df['close_smoothed'].shift(3)
        raw_df['open_lag1'] = raw_df['open'].shift(1)
        raw_df['open_lag2'] = raw_df['open'].shift(2)
        raw_df['volume_change'] = raw_df['volume'].pct_change()
        raw_df['intraday_return'] = (raw_df['close'] - raw_df['open']) / raw_df['open']
        
        bb_df = ta.bbands(raw_df['close_smoothed'], length=20, std=2)
        raw_df['bb_lower'] = bb_df.iloc[:, 0]
        raw_df['bb_middle'] = bb_df.iloc[:, 1]
        raw_df['bb_upper'] = bb_df.iloc[:, 2]
        raw_df['atr_14'] = ta.atr(raw_df['high'], raw_df['low'], raw_df['close_smoothed'], length=14)
        raw_df['ema_14'] = ta.ema(raw_df['close_smoothed'], length=14)
        raw_df['roc_10'] = ta.roc(raw_df['close_smoothed'], length=10)
        
        adx_raw = ta.adx(raw_df['high'], raw_df['low'], raw_df['close_smoothed'], length=14)
        raw_df['adx_14'] = adx_raw.iloc[:, 0]
        
        # Clean null values and calculate stationary ratios using DataTransformer
        transformer = DataTransformer(time_steps=LOOKBACK_WINDOW)
        raw_df_reset = raw_df.reset_index()
        raw_df_transformed = transformer.transform_df(raw_df_reset)
        
        recent_features = raw_df_transformed.tail(LOOKBACK_WINDOW)
        if len(recent_features) < LOOKBACK_WINDOW:
            raise HTTPException(status_code=500, detail=f"Không đủ dữ liệu {LOOKBACK_WINDOW} ngày để tạo đặc trưng")
            
        # Transform and predict
        recent_scaled = scaler_X.transform(recent_features.values)
        X_predict = recent_scaled.reshape(1, LOOKBACK_WINDOW, len(FEATURE_COLS))
        
        # Trích xuất đặc trưng lai (32 chiều ẩn từ Transformer + 22 chiều chỉ báo ngày hiện tại) cho mô hình Hybrid XGBoost
        feature_extractor = tf.keras.models.Model(
            inputs=transformer_model.input,
            outputs=transformer_model.layers[-2].output
        )
        X_predict_latent = feature_extractor.predict(X_predict, verbose=0)
        X_predict_today = X_predict[0, -1, :].reshape(1, -1)
        X_predict_hybrid = np.concatenate([X_predict_latent, X_predict_today], axis=1)

        
        xgb_pred_scaled = xgb_model.predict(X_predict_hybrid).reshape(-1, 1)
        xgb_return_future = scaler_y.inverse_transform(xgb_pred_scaled)[0][0]
        
        trans_pred_scaled = transformer_model.predict(X_predict, verbose=0)
        trans_return_future = scaler_y.inverse_transform(trans_pred_scaled)[0][0]

        
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
