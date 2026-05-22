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
    *   Cấu hình tham số `INDIVIDUAL_TRAINING = True` trong `main.py` để chạy vòng lặp huấn luyện riêng biệt cho từng mã cổ phiếu.
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

---

## 🏆 KẾT QUẢ ĐẠT ĐƯỢC SAU NÂNG CẤP

Nhờ các cải tiến trên, sai số dự báo trung bình tuyệt đối (MAE) trên tập kiểm thử đã giảm đáng kể:

1.  **Mã Vinamilk (VNM.VN):**
    *   *XGBoost:* MAE **223,30 VNĐ** (Lệch **0.37%**).
    *   *Transformer:* MAE **401,44 VNĐ** (Lệch **0.67%**).
2.  **Mã Google (GOOGL):**
    *   *XGBoost:* MAE **41.090,33 VNĐ** (~$1.62 USD, Lệch **0.85%**).
    *   *Transformer:* MAE **74.021,98 VNĐ** (~$2.91 USD, Lệch **1.52%**).
3.  **Mã Meta (META):**
    *   *XGBoost:* MAE **747.162,18 VNĐ** (~$29.42 USD, Lệch **4.89%**).
    *   *Transformer:* MAE **741.061,81 VNĐ** (~$29.18 USD, Lệch **4.95%**).

---

## 📂 DANH SÁCH CÁC FILE CHÍNH TRONG HỆ THỐNG

*   `main.py`: Điểm kích hoạt toàn bộ luồng tải dữ liệu, chuẩn hóa, huấn luyện độc lập cho từng mã, vẽ biểu đồ so sánh kết quả và đưa ra dự báo cho phiên tiếp theo.
*   `src/data_loader.py`: Tải dữ liệu từ Yahoo Finance & DNSE API, đồng nhất tiền tệ, xử lý khuyết thiếu, và tính toán các đặc trưng chỉ báo kỹ thuật.
*   `src/features.py`: Lớp `DataTransformer` phụ trách việc co giãn dữ liệu (MinMaxScaler) và tạo cửa sổ trượt 3D cho các mô hình.
*   `src/ai_models.py`: Định nghĩa kiến trúc mạng Transformer chuyên sâu và luồng GridSearchCV tối ưu XGBoost (hoàn toàn không còn LSTM).
*   `docs/PROJECT_NOTES.md`: Báo cáo chi tiết và tài liệu hướng dẫn kỹ thuật cho người dùng.
*   `notebooks/01_EDA.ipynb`: Jupyter Notebook phân tích và trực quan hóa dữ liệu khám phá.
