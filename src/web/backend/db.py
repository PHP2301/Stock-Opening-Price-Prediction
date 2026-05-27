import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Data directory relative to src/web/backend is root/data (up three levels)
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
os.makedirs(DB_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'stock_predictions.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Stock(Base):
    __tablename__ = "stocks"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    currency = Column(String, default="VND")
    
    prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    sentiments = relationship("NewsSentiment", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="stock", cascade="all, delete-orphan")

class StockPrice(Base):
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    date = Column(String, index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    stock = relationship("Stock", back_populates="prices")
    
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uix_stock_date"),)

class NewsSentiment(Base):
    __tablename__ = "news_sentiments"
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    published_date = Column(String, nullable=False)
    title = Column(String, nullable=False)
    source = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=False)
    sentiment_label = Column(String, nullable=False)

    stock = relationship("Stock", back_populates="sentiments")

class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    prediction_date = Column(String, index=True, nullable=False)
    target_date = Column(String, nullable=False)
    xgb_predicted_price = Column(Float, nullable=False)
    xgb_lower = Column(Float, nullable=False)
    xgb_upper = Column(Float, nullable=False)
    trans_predicted_price = Column(Float, nullable=False)
    trans_lower = Column(Float, nullable=False)
    trans_upper = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    usd_vnd_rate = Column(Float, nullable=True)
    actual_open_price = Column(Float, nullable=True)

    stock = relationship("Stock", back_populates="predictions")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
