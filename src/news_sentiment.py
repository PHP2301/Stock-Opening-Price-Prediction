import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import urllib.request
import xml.etree.ElementTree as ET
from deep_translator import GoogleTranslator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Đảm bảo VADER lexicon đã được tải về máy
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    def __init__(self, engine='vader'):
        self.engine = engine.lower()
        self.tokenizer = None
        self.model = None
        self.vader = None
        
        if self.engine == 'finbert':
            print("  [NLP] Đang tải mô hình FinBERT (ProsusAI/finbert) (~400MB)...")
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.torch = torch
                print("  [NLP] Tải mô hình FinBERT thành công!")
            except Exception as e:
                print(f"  [NLP] Không thể tải FinBERT: {e}. Tự động chuyển sang dùng VADER.")
                self.engine = 'vader'
                
        if self.engine == 'vader':
            self.vader = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, texts):
        """
        Phân tích cảm xúc của một danh sách tiêu đề tin tức.
        Trả về điểm số trung bình từ -1.0 (tiêu cực) đến 1.0 (tích cực).
        """
        if not texts:
            return 0.0
        
        if self.engine == 'finbert' and self.model is not None:
            try:
                # Tokenize các văn bản và dự đoán lớp cảm xúc
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                # Lấy xác suất phân loại (0: positive, 1: negative, 2: neutral)
                import torch.nn.functional as F
                probs = F.softmax(outputs.logits, dim=-1).numpy()
                # Điểm số = Xác suất tích cực * 1.0 + Xác suất tiêu cực * -1.0
                scores = probs[:, 0] * 1.0 - probs[:, 1] * 1.0
                return float(np.mean(scores))
            except Exception as e:
                print(f"  [NLP] Lỗi phân tích FinBERT: {e}. Chuyển sang VADER cho tin này.")
        
        # Fallback hoặc mặc định sử dụng VADER
        scores = []
        for text in texts:
            vs = self.vader.polarity_scores(text)
            scores.append(vs['compound'])
        return float(np.mean(scores)) if scores else 0.0

def fetch_latest_news(ticker):
    """
    Thu thập các tin tức tài chính trực tuyến mới nhất.
    """
    news_list = []
    
    # 1. Tải tin tức quốc tế từ yfinance (Bloomberg, Reuters, CNBC, Seeking Alpha...)
    try:
        yfticker = yf.Ticker(ticker)
        yf_news = yfticker.news
        if yf_news:
            for item in yf_news:
                title = item.get('title', '')
                pub_time = item.get('providerPublishTime', 0)
                if title and pub_time:
                    dt = datetime.fromtimestamp(pub_time)
                    news_list.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'title': title
                    })
    except Exception as e:
        print(f"  [NEWS] Không thể tải tin tức yfinance cho {ticker}: {e}")

    # 2. Nếu là mã Việt Nam (VNM.VN), tải thêm các tin tiếng Việt qua RSS CafeF & dịch tự động
    if "VNM" in ticker.upper():
        rss_urls = [
            "https://cafef.vn/thi-truong-chung-khoan.rss",
            "https://cafef.vn/doanh-nghiep.rss"
        ]
        try:
            translator = GoogleTranslator(source='vi', target='en')
            for url in rss_urls:
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        xml_data = response.read()
                    root = ET.fromstring(xml_data)
                    
                    items = root.findall('.//item')[:10] # Lấy tối đa 10 tin mới nhất để tối ưu tốc độ
                    vi_titles = []
                    dates = []
                    
                    for item in items:
                        title_vi = item.find('title').text
                        pub_date_str = item.find('pubDate').text
                        if title_vi and pub_date_str:
                            parts = pub_date_str.split(' ')
                            if len(parts) >= 4:
                                day = parts[1]
                                month_str = parts[2]
                                year = parts[3]
                                months = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                                          'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
                                month = months.get(month_str[:3], '01')
                                date_str = f"{year}-{month}-{day.zfill(2)}"
                                vi_titles.append(title_vi)
                                dates.append(date_str)
                    
                    if vi_titles:
                        # Dịch cả lô tiêu đề trong 1 request duy nhất
                        try:
                            en_titles = translator.translate_batch(vi_titles)
                            for date_str, title_en in zip(dates, en_titles):
                                news_list.append({
                                    'date': date_str,
                                    'title': f"[VN] {title_en}"
                                })
                        except Exception as e:
                            # Fallback nếu dịch theo lô bị lỗi
                            for date_str, title_vi in zip(dates, vi_titles):
                                news_list.append({
                                    'date': date_str,
                                    'title': f"[VN-Raw] {title_vi}"
                                })
                except Exception as e:
                    pass
        except Exception as e:
            print(f"  [NEWS] Không thể dịch tin tức Việt Nam: {e}")
            
    return news_list

def get_news_sentiment_features(ticker, dates_list, engine='vader'):
    """
    Trả về DataFrame chứa đặc trưng 'sentiment_score' và 'news_volume' cho danh sách các ngày yêu cầu.
    """
    cache_path = os.path.join('data', f'{ticker}_sentiment_cache.csv')
    df_cache = pd.DataFrame(columns=['date', 'sentiment_score', 'news_volume'])
    
    if os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path)
            df_cache['date'] = df_cache['date'].astype(str)
        except Exception as e:
            pass

    # Lấy danh sách ngày độc nhất để chuẩn bị cập nhật tin tức trực tuyến
    dates_list = [str(d)[:10] for d in dates_list]
    
    # Thu thập tin trực tuyến mới nhất
    print(f"  [NEWS] Đang tải và phân tích cảm xúc tin tức mới nhất cho {ticker}...")
    latest_news = fetch_latest_news(ticker)
    
    if latest_news:
        analyzer = SentimentAnalyzer(engine=engine)
        # Gộp các tin tức theo ngày để phân tích
        news_by_date = {}
        for item in latest_news:
            d = item['date']
            news_by_date.setdefault(d, []).append(item['title'])
            
        new_records = []
        for d, titles in news_by_date.items():
            score = analyzer.analyze_sentiment(titles)
            new_records.append({
                'date': d,
                'sentiment_score': score,
                'news_volume': len(titles)
            })
            
        df_new = pd.DataFrame(new_records)
        # Hợp nhất với cache cũ (ưu tiên bản ghi mới nhất)
        if not df_cache.empty:
            df_merged = pd.concat([df_cache, df_new]).drop_duplicates(subset=['date'], keep='last')
        else:
            df_merged = df_new
        
        # Lưu cache mới
        df_merged.to_csv(cache_path, index=False)
        df_cache = df_merged

    # Tạo DataFrame đầy đủ cho toàn bộ dòng thời gian lịch sử
    df_all_dates = pd.DataFrame({'date': dates_list})
    df_all_dates['date'] = df_all_dates['date'].astype(str)
    
    if not df_cache.empty:
        df_features = pd.merge(df_all_dates, df_cache, on='date', how='left')
    else:
        df_features = df_all_dates.copy()
        df_features['sentiment_score'] = np.nan
        df_features['news_volume'] = np.nan

    # Xử lý các ngày không có tin tức (bao gồm các ngày giao dịch lịch sử cũ)
    # Các ngày lịch sử cũ chưa có tin tức sẽ được đặt mặc định điểm cảm xúc = 0 (trung lập)
    df_features['sentiment_score'] = df_features['sentiment_score'].fillna(0.0)
    df_features['news_volume'] = df_features['news_volume'].fillna(0.0)
    
    # Áp dụng nội suy xu hướng ngắn hạn cho các ngày xen kẽ
    df_features['sentiment_score'] = df_features['sentiment_score'].ffill().bfill()
    df_features['news_volume'] = df_features['news_volume'].ffill().bfill()
    
    return df_features[['date', 'sentiment_score', 'news_volume']]

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    # Test thử module độc lập
    print("=== TEST TẢI TIN TỨC & PHÂN TÍCH CẢM XÚC ===")
    dates = ["2026-05-22", "2026-05-23", "2026-05-24", "2026-05-25"]
    df_res = get_news_sentiment_features("META", dates, engine='vader')
    print(df_res)
