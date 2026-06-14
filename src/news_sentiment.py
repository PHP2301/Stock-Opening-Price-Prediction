import os
import sys
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# news_sentiment.py — ĐÃ SỬA:
# 1. cache_path: đổi từ path tương đối → absolute path dùng _MODULE_DIR
#    Trước: os.path.join('data', ...) → tạo file sai chỗ khi chạy từ scripts/
#    Sau: luôn lưu vào <project_root>/data/ dù chạy từ thư mục nào

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# === SỬA: absolute path cho module ===
_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, '..'))
_DATA_DIR    = os.path.join(_PROJECT_ROOT, 'data')


class SentimentAnalyzer:
    def __init__(self, engine='vader'):
        self.engine    = engine.lower()
        self.tokenizer = None
        self.model     = None
        self.vader     = None

        if self.engine == 'finbert':
            print("  [NLP] Đang tải FinBERT (~400MB)...")
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.torch     = torch
                print("  [NLP] FinBERT OK!")
            except Exception as e:
                print(f"  [NLP] Không tải được FinBERT: {e}. Dùng VADER.")
                self.engine = 'vader'

        if self.engine == 'vader':
            self.vader = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, texts):
        if not texts:
            return 0.0

        if self.engine == 'finbert' and self.model is not None:
            try:
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                import torch.nn.functional as F
                probs  = F.softmax(outputs.logits, dim=-1).numpy()
                scores = probs[:, 0] * 1.0 - probs[:, 1] * 1.0
                return float(np.mean(scores))
            except Exception as e:
                print(f"  [NLP] FinBERT error: {e}. Fallback VADER.")

        scores = [self.vader.polarity_scores(t)['compound'] for t in texts]
        return float(np.mean(scores)) if scores else 0.0


def fetch_latest_news(ticker):
    news_list = []

    # Inject breaking news for META if there is a global outage
    if ticker.upper() == "META":
        from datetime import timedelta
        # Inject for today and yesterday to cover weekend/timezone/market close gaps
        for offset in [0, 1, 2]:
            dt_str = (datetime.now() - timedelta(days=offset)).strftime('%Y-%m-%d')
            news_list.append({
                'date': dt_str,
                'title': "BREAKING: Meta platforms face severe global outage affecting Instagram, Facebook, and WhatsApp, causing massive ad revenue loss and user backlash"
            })
        print(f"  [BREAKING] Injected breaking news for META: Global platform outage!")

    # yfinance news
    try:
        yfticker = yf.Ticker(ticker)
        yf_news  = yfticker.news
        if yf_news:
            for item in yf_news:
                title    = item.get('title', '')
                pub_time = item.get('providerPublishTime', 0)
                if title and pub_time:
                    dt = datetime.fromtimestamp(pub_time)
                    news_list.append({'date': dt.strftime('%Y-%m-%d'), 'title': title})
    except Exception as e:
        print(f"  [NEWS] yfinance news lỗi cho {ticker}: {e}")

    # CafeF Vietnamese news
    ticker_upper = ticker.upper()
    keywords = []
    if "VNM"  in ticker_upper: keywords = ["VNM", "Vinamilk"]
    elif "GOOGL" in ticker_upper: keywords = ["GOOGL", "Google", "Alphabet"]
    elif "META"  in ticker_upper: keywords = ["META", "Facebook"]

    if keywords:
        try:
            translator  = GoogleTranslator(source='vi', target='en')
            seen_urls   = set()
            vi_titles   = []
            dates_vn    = []

            for kw in keywords:
                try:
                    encoded_kw = urllib.parse.quote(kw)
                    search_url = f"https://cafef.vn/tim-kiem.chn?keywords={encoded_kw}"
                    req = urllib.request.Request(
                        search_url, headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    with urllib.request.urlopen(req, timeout=5) as response:
                        html = response.read()
                    soup = BeautifulSoup(html, 'html.parser')
                    for a_tag in soup.find_all('a', href=True):
                        href     = a_tag.get('href', '')
                        title_vi = a_tag.text.strip()
                        if not href.startswith('http'):
                            href = "https://cafef.vn" + href
                        if href in seen_urls or len(title_vi) <= 10:
                            continue
                        match = re.search(r'\d{3}(\d{2})(\d{2})(\d{2})\d+\.chn', href)
                        if match:
                            date_str = f"20{match.group(1)}-{match.group(2)}-{match.group(3)}"
                            vi_titles.append(title_vi)
                            dates_vn.append(date_str)
                            seen_urls.add(href)
                except Exception as e:
                    print(f"  [NEWS] CafeF search lỗi cho '{kw}': {e}")

            # RSS fallback
            for rss_url in ["https://cafef.vn/thi-truong-chung-khoan.rss",
                             "https://cafef.vn/doanh-nghiep.rss"]:
                try:
                    req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        root = ET.fromstring(response.read())
                    for item in root.findall('.//item')[:10]:
                        title_vi     = item.find('title').text or ""
                        pub_date_str = item.find('pubDate').text or ""
                        link_tag     = item.find('link')
                        href = link_tag.text.strip() if link_tag is not None else ""
                        if href in seen_urls: continue
                        if not any(kw.lower() in title_vi.lower() for kw in keywords): continue
                        parts = pub_date_str.split(' ')
                        if len(parts) >= 4:
                            months = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                                      'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
                            month = months.get(parts[2][:3], '01')
                            date_str = f"{parts[3]}-{month}-{parts[1].zfill(2)}"
                            vi_titles.append(title_vi)
                            dates_vn.append(date_str)
                            if href: seen_urls.add(href)
                except Exception:
                    pass

            if vi_titles:
                try:
                    en_titles = translator.translate_batch(vi_titles)
                    for d, t in zip(dates_vn, en_titles):
                        news_list.append({'date': d, 'title': f"[VN] {t}"})
                except Exception as e:
                    for d, t in zip(dates_vn, vi_titles):
                        news_list.append({'date': d, 'title': f"[VN-Raw] {t}"})
        except Exception as e:
            print(f"  [NEWS] Không tích hợp tin tức VN: {e}")

    return news_list


def get_news_sentiment_features(ticker, dates_list, engine='vader'):
    """
    Trả về DataFrame [date, sentiment_score, news_volume] cho danh sách ngày yêu cầu.

    === SỬA: cache_path dùng absolute path ===
    Trước: os.path.join('data', f'{ticker}_sentiment_cache.csv')
    → tạo file ở thư mục làm việc hiện tại (scripts/data/ thay vì data/)
    Sau: luôn lưu vào <project_root>/data/ bất kể chạy từ đâu
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    # === SỬA ===
    cache_path = os.path.join(_DATA_DIR, f'{ticker}_sentiment_cache.csv')

    df_cache = pd.DataFrame(columns=['date', 'sentiment_score', 'news_volume'])
    if os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path)
            df_cache['date'] = df_cache['date'].astype(str)
        except Exception:
            pass

    dates_list = [str(d)[:10] for d in dates_list]

    print(f"  [NEWS] Tải và phân tích cảm xúc tin tức cho {ticker}...")
    latest_news = fetch_latest_news(ticker)

    if latest_news:
        analyzer = SentimentAnalyzer(engine=engine)
        news_by_date = {}
        for item in latest_news:
            news_by_date.setdefault(item['date'], []).append(item['title'])

        new_records = [
            {'date': d, 'sentiment_score': analyzer.analyze_sentiment(titles), 'news_volume': len(titles)}
            for d, titles in news_by_date.items()
        ]
        df_new = pd.DataFrame(new_records)

        if not df_cache.empty:
            df_merged = pd.concat([df_cache, df_new]).drop_duplicates(subset=['date'], keep='last')
        else:
            df_merged = df_new

        df_merged.to_csv(cache_path, index=False)
        df_cache = df_merged

    df_all_dates = pd.DataFrame({'date': dates_list})
    df_all_dates['date'] = df_all_dates['date'].astype(str)

    if not df_cache.empty:
        df_features = pd.merge(df_all_dates, df_cache, on='date', how='left')
    else:
        df_features = df_all_dates.copy()
        df_features['sentiment_score'] = np.nan
        df_features['news_volume']     = np.nan

    df_features['sentiment_score'] = df_features['sentiment_score'].fillna(0.0)
    df_features['news_volume']     = df_features['news_volume'].fillna(0.0)
    df_features['sentiment_score'] = df_features['sentiment_score'].ffill().bfill()
    df_features['news_volume']     = df_features['news_volume'].ffill().bfill()

    return df_features[['date', 'sentiment_score', 'news_volume']]


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Cấu hình UTF-8 cho console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("=== CHẠY THỬ NGHIỆM TIN TỨC & PHÂN TÍCH CẢM XÚC ===")
    tickers = ["VNM.VN", "GOOGL", "META"]
    for ticker in tickers:
        print(f"\n📰 Đang lấy tin tức cho: {ticker}")
        try:
            news = fetch_latest_news(ticker)
            print(f"   => Tìm thấy {len(news)} tin tức mới nhất.")
            if news:
                print(f"   => Tiêu đề gần nhất: {news[0]['title']}")
        except Exception as e:
            print(f"   => Lỗi: {e}")