# Nhật Ký Lịch Sử Dự Án: Stock Opening Price Prediction

Tài liệu này ghi lại toàn bộ lộ trình phát triển, các khó khăn kỹ thuật đã vượt qua, và các dấu mốc quan trọng đạt được trong quá trình xây dựng hệ thống dự báo giá mở cửa cho 3 mã cổ phiếu **VNM.VN**, **GOOGL**, và **META**.

---

## 📅 LỊCH TRÌNH VÀ TIẾN ĐỘ THỰC HIỆN

### 🔴 Ngày 1 & 2: Khảo Sát & Thiết Lập Hệ Thống Ban Đầu
*   **Mục tiêu:** Thiết lập nền tảng dự án và thực hiện EDA (Phân tích dữ liệu khám phá).
*   **Công việc đã làm:**
    *   Tạo cấu trúc thư mục chuẩn cho dự án ML/DL (`src/`, `models/`, `data/`, `notebooks/`).
    *   Hoàn thành file notebook phân tích dữ liệu `notebooks/01_EDA.ipynb`.
    *   Xác định các chỉ báo kỹ thuật cơ bản cần dùng: SMA, MACD, RSI, Bollinger Bands.
    *   Tải dữ liệu từ Yahoo Finance cho cả 3 mã.
*   **Vấn đề gặp phải:** Thư viện `pandas_ta` gặp lỗi không tìm thấy (ModuleNotFoundError) khi chạy trong Jupyter Notebook.
*   **Giải pháp:** Cấu hình lại Python Kernel để sử dụng đúng thư mục môi trường ảo `.venv/` chứa đầy đủ dependencies.

### 🟡 Ngày 3: Tích Hợp Đa Mã & Đồng Nhất Đơn Vị Tệ
*   **Mục tiêu:** Xây dựng luồng huấn luyện chung (multi-ticker training) để tận dụng dữ liệu chéo.
*   **Công việc đã làm:**
    *   Viết mã nguồn `src/data_loader.py` để tải và làm sạch dữ liệu.
    *   Đồng nhất đơn vị tiền tệ: Quy đổi toàn bộ giá của **GOOGL** và **META** sang **VNĐ** với tỷ giá tham chiếu cố định `1 USD = 25.400 VNĐ`.
    *   Viết lớp `DataTransformer` trong `src/features.py` để tạo các sliding windows (cửa sổ trượt) dạng 3D chuẩn bị cho mạng Neural và XGBoost.
    *   Huấn luyện các mô hình ban đầu: **XGBoost**, **LSTM**, **Transformer**, và **Ensemble (kết hợp LSTM + XGBoost)**.
*   **Kết quả ban đầu:** Mô hình Ensemble đạt kết quả tốt nhất. Tuy nhiên, sai số dự báo của Vinamilk vẫn ở mức khá cao và các mã nước ngoài gặp sai số lớn do biến động tỷ giá và sự khác biệt về cấu trúc giá.

### 🟢 Ngày 4: Chuyển Đổi Chiến Lược & Loại Bỏ LSTM/Ensemble
*   **Mục tiêu:** Tối ưu hóa sâu theo chỉ đạo của người dùng: loại bỏ hoàn toàn LSTM và Ensemble, chỉ giữ lại XGBoost và Transformer, đồng thời chuyển sang huấn luyện độc lập cho từng mã (Individual Training) để tăng độ chính xác.
*   **Công việc đã làm:**
    *   Cấu hình tham số `INDIVIDUAL_TRAINING = True` trong `scripts/run_pipeline.py` để chạy vòng lặp huấn luyện riêng biệt cho từng mã cổ phiếu.
    *   Loại bỏ hoàn toàn kiến trúc LSTM và khối Ensemble khỏi luồng xử lý, đánh giá và cấu trúc file nguồn `src/ai_models.py`.
    *   Phát hiện nguyên nhân Transformer có kết quả kém hơn: lượng dữ liệu huấn luyện quá ít (dữ liệu Vinamilk trường thực tế chỉ có từ tháng 9/2019 đến nay, khoảng ~1.600 phiên).

### 🔵 Ngày 5: Nâng Cấp Toàn Diện & Giải Quyết Lỗi Hệ Thống
*   **Mục tiêu:** Mở rộng dữ liệu lịch sử trước 2019 cho Vinamilk, nâng cấp sâu kiến trúc mạng Transformer & XGBoost, và tối ưu hóa thời gian chạy.
*   **Công việc đã làm:**
    *   **Mở rộng dữ liệu:** Tích hợp **DNSE Chart API** tải bổ sung **1.868 phiên giao dịch lịch sử** của VNM.VN trước năm 2019 (từ 2012 đến 2019), nâng tổng số phiên huấn luyện lên **3.532 phiên** (~14 năm dữ liệu).
    *   **Nâng cấp Transformer:** Tăng Lookback Window từ 30 lên **45 ngày**. Sửa đổi hàm `build_transformer` thành kiến trúc sâu hơn: **2 lớp Attention**, **8 heads**, **key_dim=128**, tích hợp các kết nối tắt (Residual Connections) và Layer Normalization để tránh suy giảm gradient.
    *   **Nâng cấp XGBoost:** Bổ sung 3 đặc trưng động lượng nâng cao: **EMA_14** (Đường trung bình di động lũy thừa), **ROC_10** (Tốc độ thay đổi giá), và **ADX_14** (Chỉ số định hướng trung bình).
    *   **Tối ưu hóa GridSearchCV:** Mở rộng danh mục tìm kiếm siêu tham số cho XGBoost (`colsample_bytree`, `subsample`, `max_depth`, `learning_rate`, `n_estimators`). Thiết lập chạy song song `n_jobs=-1` để rút ngắn thời gian tìm kiếm từ 12 phút xuống còn **1 phút**.
    *   **Giải quyết lỗi Treo Hệ thống:** Sửa lỗi Matplotlib GUI bị treo vô hạn trên Windows console bằng cách ép buộc Matplotlib chạy ở chế độ headless (`matplotlib.use('Agg')`).
    *   **Khắc phục lỗi Unicode trên Console:** Cấu hình lại chuẩn mã hóa của console (`sys.stdout.reconfigure(encoding='utf-8')`) để hiển thị thông tin và log tiếng Việt hoàn hảo trên Windows PowerShell/CMD.

### 🟣 Ngày 6: Tích Hợp 3 Nâng Cấp Sâu & Hệ Thống Hóa Kiến Trúc
*   **Mục tiêu:** Tích hợp đồng thời 3 nâng cấp chuyên sâu (Market Index, Conv1D-Transformer, Cosine Decay) và cung cấp tài liệu kiến trúc toàn diện từ đầu đến cuối.
*   **Công việc đã làm:**
    *   **Tích hợp chỉ số thị trường vĩ mô (`market_return`):** Tự động tải chỉ số ETF vĩ mô đại diện thị trường Việt Nam (`VNM`) và chỉ số Mỹ (`^GSPC`) để tính tỷ suất sinh lời thị trường và đưa vào làm đặc trưng đầu vào thứ 15.
    *   **Mạng lai ghép Conv1D-Transformer:** Thêm lớp trích xuất đặc trưng cục bộ `Conv1D` trước Positional Embedding để tối ưu hóa khả năng nắm bắt chuyển động chuỗi thời gian ngắn hạn.
    *   **Hạ tốc độ học Cosine Decay:** Thay thế `ReduceLROnPlateau` bằng callback `LearningRateScheduler` suy giảm Cosine giúp mạng Neural hội tụ tối ưu hơn.
    *   **Hệ thống hóa tài liệu kiến trúc:** Tạo folder mới và viết file `docs/explain/system_overview.md` để giải thích cặn kẽ nguyên lý hoạt động của toàn bộ hệ thống từ đầu đến cuối cho người dùng.

### 🟤 Ngày 7 (29/05/2026): Lọc Tin Tức Theo Mã Cổ Phiếu & Tích Hợp Cơ Sở Dữ Liệu
*   **Mục tiêu:** Khử hoàn toàn tin tức ngoài luồng (như MCM) và lưu trữ tin tức vào DB để hiển thị lên bảng tin Web App.
*   **Công việc đã làm:**
    *   **Phân rã từ khóa động**: Lập cấu hình từ khóa riêng cho VNM.VN (`VNM`, `Vinamilk`), GOOGL (`Google`, `Alphabet`), và META (`Facebook`) để lọc chính xác tin tức liên quan.
    *   **Tích hợp SQLite DB cho News**: Sửa đổi `src/web/backend/api.py` để tự động phân tích điểm cảm xúc và lưu trữ từng bài viết vào bảng `news_sentiments` khi kích hoạt dự báo.
    *   **Làm sạch bộ nhớ đệm (Cache)**: Sửa lỗi hiển thị năm 2 chữ số trong Cache và tái tạo lại các cache file sạch cho VNM.VN và META.

---

## 🏆 KẾT QUẢ ĐẠT ĐƯỢC SAU NÂNG CẤP

Nhờ các cải tiến trên, sai số dự báo trung bình tuyệt đối (MAE) trên tập kiểm thử đã giảm đáng kể:

1.  **Mã Vinamilk (VNM.VN):**
    *   *XGBoost:* MAE **224,26 VNĐ** (Lệch **0.37%**).
    *   *Transformer:* MAE **475,02 VNĐ** (Lệch **0.79%**).
2.  **Mã Google (GOOGL):**
    *   *XGBoost:* MAE **40.256,13 VNĐ** (~$1.58 USD, Lệch **0.83%**).
    *   *Transformer:* MAE **101.113,12 VNĐ** (~$3.98 USD, Lệch **1.87%**).
3.  **Mã Meta (META):**
    *   *XGBoost:* MAE **640.288,60 VNĐ** (~$25.21 USD, Lệch **4.20%**).
    *   *Transformer:* MAE **1.016.060,17 VNĐ** (~$40.00 USD, Lệch **7.18%**).

---


### 🔵 Ngày 8 & 9 (12/06/2026 - 13/06/2026): Triển khai Rolling Walk-Forward Backtesting (Phase 2.3)
*   **Mục tiêu:** Xây dựng hệ thống tự động tái huấn luyện cuốn chiếu (Rolling retraining) theo năm để thích ứng với hiện tượng trượt phân phối dữ liệu (Distribution Shift) và so sánh hiệu quả thực tế.
*   **Công việc đã làm:**
    *   Phát triển kịch bản `scripts/run_walk_forward_backtest.py` chạy qua 3 Window cuốn chiếu (2023, 2024, 2025-2026).
    *   Tách biệt khâu fit/transform của Scaler trên tập huấn luyện của từng Window nhằm chống rò rỉ thông tin tương lai.
    *   Thiết kế cơ chế nạp trọng số warm-start (set_weights) cho Transformer qua từng Window giúp rút ngắn thời gian huấn luyện từ 3 phút xuống dưới 30 giây.
    *   Thực hiện chạy và đánh giá hiệu quả trên cả 3 mã với threshold 0.10% và lưu trữ các đồ thị Equity Curve tương ứng.
*   **Kết quả:**
    *   Mô hình Rolling giải quyết xuất sắc bias âm của GOOGL trong giai đoạn AI bull run, giúp chiến lược đạt lợi nhuận **+100.75%** (Sharpe 1.39) so với mô hình tĩnh.
    *   Đối với VNM.VN, tỷ lệ thắng vẫn ở mức rất thấp (12.82%), củng cố kết luận rằng VNM.VN không có Edge thực sự để giao dịch hiệu quả.

## 📂 DANH SÁCH CÁC FILE CHÍNH TRONG HỆ THỐNG

*   `scripts/run_pipeline.py`: Điểm kích hoạt toàn bộ luồng tải dữ liệu, chuẩn hóa, huấn luyện độc lập cho từng mã, vẽ biểu đồ so sánh kết quả và đưa ra dự báo cho phiên tiếp theo.
*   `src/data_loader.py`: Tải dữ liệu từ Yahoo Finance & DNSE API, đồng nhất tiền tệ, xử lý khuyết thiếu, và tính toán các đặc trưng chỉ báo kỹ thuật.
*   `src/features.py`: Lớp `DataTransformer` phụ trách việc co giãn dữ liệu (MinMaxScaler) và tạo cửa sổ trượt 3D cho các mô hình.
*   `src/ai_models.py`: Định nghĩa kiến trúc mạng Transformer chuyên sâu và luồng GridSearchCV tối ưu XGBoost (hoàn toàn không còn LSTM).
*   `docs/explain/system_overview.md`: Tài liệu giải thích chi tiết toàn bộ nguyên lý hoạt động và luồng đi của dữ liệu từ đầu đến cuối của dự án.
*   `docs/PROJECT_NOTES.md`: Báo cáo chi tiết và tài liệu hướng dẫn kỹ thuật cho người dùng.
*   `notebooks/01_EDA.ipynb`: Jupyter Notebook phân tích và trực quan hóa dữ liệu khám phá.
