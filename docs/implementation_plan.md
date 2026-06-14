# Kế hoạch Triển khai Tổng hợp: Hệ thống Dự báo Giá mở cửa & Multi-Agent

Tài liệu này tổng hợp toàn bộ các kết quả thực tế của hệ thống, dọn dẹp các mục đã hoàn thành, chỉ ra các lỗ hổng cần làm sạch, và đề xuất lộ trình chi tiết để tiếp tục thực hiện.

---

## 🏆 ĐÁNH GIÁ TRẠNG THÁI THỰC TẾ & KẾT QUẢ ĐÃ ĐẠT ĐƯỢC

Sau khi kiểm tra toàn bộ mã nguồn hiện tại, dưới đây là tình trạng thực tế của các mục tiêu:

### 1. Dữ liệu & Đặc trưng (Feature Engineering) — [HOÀN THÀNH]
- **Tích hợp 42 đặc trưng High-Alpha**: Đã tích hợp thành công các chỉ báo nâng cao (`mfi_14`, các chỉ báo đếm ngày sự kiện cổ tức `dividend_flag`, `days_to_dividend`, `days_after_dividend` lấy từ yfinance thực tế qua `dividend_fetcher.py`, và các proxy dòng tiền ngoại/tự doanh).
- **Khử rò rỉ dữ liệu**: Đồng bộ múi giờ Mỹ-Việt (dịch chuyển `shift(1)` đối với dữ liệu vĩ mô Mỹ cho mã Việt Nam) và loại bỏ hoàn toàn các tỷ lệ giá tương lai phi tĩnh khỏi tập đặc trưng đầu vào.

### 2. Loại bỏ XGBoost khỏi Hệ thống — [HOÀN THÀNH MỘT PHẦN]
- **Huấn luyện & Dự báo**: Đã loại bỏ hoàn toàn XGBoost khỏi `run_training.py` và `predict.py`, sử dụng Transformer thuần túy làm mô hình cốt lõi duy nhất.
- **Lỗ hổng (Cần sửa đổi)**: Hai tệp kiểm thử lịch sử `run_backtest.py` và `run_walk_forward_backtest.py` vẫn gọi và yêu cầu mô hình XGBoost theo mặc định.
  - *Giải pháp*: Cấu hình trực tiếp biến mặc định `trans_only = True` trong code của cả 2 file. Xóa/comment đoạn tải tệp mô hình XGBoost (`xgb_path` và `joblib.load`) để tránh các cảnh báo lỗi không đáng có, thay vào đó gán cứng `xgb_model = None`.

### 3. Hệ thống Multi-Agent & Backtesting — [HOÀN THÀNH MỘT PHẦN]
- **Backtesting**: Đã bổ sung các hệ số Sharpe, Max Drawdown, Calmar và vẽ biểu đồ Equity Curve kết hợp hệ số trượt giá động (Dynamic Slippage).
- **Kiến trúc Multi-Agent**: Xây dựng thành công các Agent (Technical, Sentiment, Macro, Risk) và Orchestrator điều phối tranh luận Bull vs Bear.
- **Lỗ hổng (Cần sửa đổi)**: Thuật toán quản lý vị thế động Kelly Criterion hiện đang được viết trực tiếp tại `scripts/predict.py` chứ chưa được đưa vào `RiskAgent`.
  - *Giải pháp*: Chuyển toàn bộ logic tính toán Kelly Criterion thành một phương thức chuyên biệt trong `src/agents/risk_agent.py`.
  - *Định nghĩa xác suất thắng $p$*: Để tránh việc AI Agent bị quá tự tin (overconfident), chúng tôi đề xuất tính toán $p$ an toàn bằng cách lấy giá trị nhỏ nhất giữa điểm số tự tin (`confidence_score`) của Orchestrator và tỷ lệ thắng lịch sử (`win_rate_history`):
    $$p = \min(\text{confidence\_score}, \text{win\_rate\_history})$$
    - Trong backtest: `win_rate_history` được tính toán động dựa trên tỷ lệ thắng của tối đa 20 lệnh gần nhất trong danh sách mô phỏng giao dịch.
    - Trong dự báo thời gian thực (`predict.py`): `win_rate_history` sẽ được đọc từ kết quả lưu của file cấu hình hiệu suất backtest (`config/performance_metrics_{ticker}.json`). Nếu không tìm thấy, hệ thống sẽ sử dụng giá trị mặc định an toàn là **0.50** (50%).

### 4. Thông báo & Cảnh báo Telegram — [HOÀN THÀNH MỘT PHẦN]
- **Tình trạng hiện tại**: Đã cấu hình Telegram Bot gửi báo cáo HTML dự báo 3 ngày và khuyến nghị Multi-Agent.
- **Lỗ hổng (Cần sửa đổi)**: Logic cảnh báo biến động mạnh (khi tỷ lệ rủi ro ATR $\ge 3\%$ hoặc dự báo thay đổi $\ge 3\%$) đã được viết, nhưng do ngưỡng này khá cao nên ở điều kiện thị trường bình thường, người dùng chỉ nhận được thông báo 3 ngày thông thường mà không thấy tiêu đề cảnh báo.
  - *Giải pháp*: Bổ sung log chi tiết in ra console thông số kiểm tra biến động thực tế (ATR %, Forecast %) so với ngưỡng để người dùng dễ theo dõi, đồng thời cung cấp tùy chọn điều chỉnh hoặc kiểm thử tính năng này.

---

## 🚀 LỘ TRÌNH CHI TIẾT THỰC HIỆN TIẾP THEO

### ✅ Bước 1: Hoàn thiện Optuna Hyperparameter Tuning cho Transformer thuần [HOÀN THÀNH]
- **Mục tiêu**: Tối ưu hóa cấu hình mạng Transformer (gồm cả biến `key_dim`) để đạt sai số tốt nhất.
- **Proposed Changes**:
  - **`scripts/run_tuning.py`**:
    - Thêm biến `key_dim` (8, 16, 32) vào không gian tìm kiếm hyperparameter của Optuna trong hàm `objective(trial)`.
    - Đảm bảo `build_transformer` được gọi truyền đúng tham số `key_dim`.
    - Ghi nhận và lưu tham số `key_dim` tối ưu vào file cấu hình JSON (`best_transformer_params_{ticker}.json`).
  - **`scripts/run_training.py`**:
    - Đọc tham số `key_dim` từ file JSON tối ưu và truyền vào `build_transformer` trong cả hai pha huấn luyện (Phase 1 & Phase 2).

### ✅ Bước 2: Module hóa Kelly Criterion, Dọn dẹp Backtest & Cấu hình Cảnh báo Volatility động [HOÀN THÀNH]
- **Mục tiêu**: Tổ chức lại mã nguồn Agent sạch sẽ, chuyển các file backtest sang chạy Transformer thuần mặc định, và cấu hình ngưỡng cảnh báo động theo ticker.
- **Proposed Changes**:
  - **`src/agents/risk_agent.py`**:
    - Định nghĩa phương thức `calculate_position_size(self, confidence_score: float, win_rate_history: float, close_price: float, stop_loss: float, take_profit: float) -> dict` thực thi công thức Kelly Criterion rút gọn, áp dụng Half-Kelly và Hard Cap 25% phân bổ vốn với:
      $$p = \min(\text{confidence\_score}, \text{win\_rate\_history})$$
  - **`scripts/predict.py`**:
    - Gọi phương thức Kelly từ `RiskAgent`. Tự động tải `win_rate_history` từ `config/performance_metrics_{ticker}.json` (mặc định 0.50 nếu thiếu).
    - Triển khai ngưỡng cảnh báo biến động động theo từng mã (ticker-specific):
      ```python
      ALERT_THRESHOLD = {
          'VNM.VN': 0.03,   # 3.0%
          'GOOGL':  0.025,  # 2.5%
          'META':   0.025,  # 2.5%
      }
      ```
    - Thêm log in ra console thông tin debug chi tiết: `🔍 [Telegram] Kiểm tra biến động mạnh cho {ticker}: ATR = X.XX% (ngưỡng Z%), Dự báo thay đổi lớn nhất = Y.YY% (ngưỡng Z%).` để dễ theo dõi và xác nhận logic chạy bình thường.
  - **`scripts/run_backtest.py` & `scripts/run_walk_forward_backtest.py`**:
    - Thiết lập giá trị mặc định `trans_only = True` ngay trong code.
    - Xóa/comment phần load mô hình XGBoost (`xgb_path` và `joblib.load`) để tránh cảnh báo lỗi không tìm thấy file, gán thẳng `xgb_model = None`.
    - Sau khi hoàn thành backtest, tự động lưu tỷ lệ thắng tổng thể (overall win rate) của chiến lược vào file `config/performance_metrics_{ticker}.json` để `predict.py` làm cơ sở tham chiếu.

### ✅ Bước 3: Phát triển Web App Dashboard trực quan [HOÀN THÀNH]
- **Mục tiêu**: Hiển thị trực quan hóa kết quả dự đoán, lịch sử kiểm thử và nhật ký tranh luận Agent.
- **Proposed Changes**:
  - Hoàn thiện giao diện HTML/CSS/JS tại thư mục `src/web` để hiển thị biểu đồ so sánh dự báo 3 ngày vs thực tế, bảng tin tức kèm điểm cảm xúc, và hộp hội thoại tranh luận Agent (Dark/Light mode cao cấp).

### ✅ Bước 4: Tích hợp Bộ cào tin tức tiếng Việt dạng RSS hợp lệ (CafeF RSS) [HOÀN THÀNH]
- **Mục tiêu**: Thu thập tin tức tiếng Việt real-time một cách an toàn, tuân thủ điều khoản dịch vụ (ToS) của CafeF để cập nhật cảm xúc cho `VNM.VN`.
- **Proposed Changes**:
  - Thay vì cào trực tiếp trang web CafeF bằng BeautifulSoup có nguy cơ vi phạm ToS, module sẽ sử dụng nguồn **RSS Feed chính thức của CafeF**: `https://cafef.vn/rss/chung-khoan.rss` (Lựa chọn B).
  - Đối với các mã chứng khoán Mỹ (GOOGL, META), sử dụng nguồn RSS miễn phí của **Google News**: `https://news.google.com/rss/search?q={ticker}` (Lựa chọn C) để thu thập tin tức một cách hợp lệ và an toàn.
  - Lọc tin tức theo từ khóa ticker chính xác để loại bỏ nhiễu và phân tích điểm cảm xúc tích hợp vào `SentimentAgent`.

---

## 🙋‍♂️ YÊU CẦU XÁC NHẬN TỪ NGƯỜI DÙNG (USER REVIEW REQUIRED)

> [!IMPORTANT]
> Vui lòng xác nhận các điểm sau để chúng tôi tiến hành thực hiện:
> 1. **Dọn dẹp Backtest**: Bạn đã đồng ý với giải pháp đổi default `trans_only = True` và gán cứng `xgb_model = None`.
> 2. **Kiến trúc Kelly & Xác suất $p$**: Bạn đã đồng ý với phương án sử dụng $p = \min(\text{confidence\_score}, \text{win\_rate\_history})$ và lưu/tải `win_rate_history` qua file cấu hình `performance_metrics_{ticker}.json`.
> 3. **Cấu hình biến động & Tin tức**: Bạn đã đồng ý cấu hình ngưỡng cảnh báo biến động động theo ticker (`ALERT_THRESHOLD`), bổ sung debug log lên console của `predict.py`, và sử dụng nguồn **RSS Feed (CafeF & Google News)** để cào tin tức an toàn hợp lệ (không vi phạm ToS).
