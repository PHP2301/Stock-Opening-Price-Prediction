import os
import urllib.request
import urllib.error
import json
import sys

def safe_print(msg):
    enc = sys.stdout.encoding or 'utf-8'
    try:
        print(msg.encode(enc, errors='replace').decode(enc))
    except Exception:
        # Fallback to ascii representation if printing fails
        print(msg.encode('ascii', errors='replace').decode('ascii'))

def load_dotenv():
    """
    Tải các biến môi trường từ file .env ở thư mục gốc (nếu có)
    mà không cần thư viện python-dotenv.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip('"').strip("'")

def send_telegram_message(text: str):
    """
    Gửi tin nhắn HTML tới Telegram Chat qua Bot API.
    """
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        safe_print("⚠️ [Telegram] Chưa cấu hình TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID trong môi trường hoặc file .env")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if res_data.get("ok"):
                safe_print("✅ [Telegram] Gửi thông báo qua Telegram thành công!")
                return True
            else:
                safe_print(f"❌ [Telegram] Phản hồi lỗi từ Telegram API: {res_data}")
                return False
    except urllib.error.HTTPError as he:
        try:
            err_body = he.read().decode("utf-8")
            safe_print(f"❌ [Telegram] Lỗi HTTP {he.code}: {he.reason}\nChi tiết: {err_body}")
        except Exception:
            safe_print(f"❌ [Telegram] Lỗi HTTP {he.code}: {he.reason}")
        return False
    except Exception as e:
        safe_print(f"❌ [Telegram] Lỗi kết nối Telegram API: {e}")
        return False
