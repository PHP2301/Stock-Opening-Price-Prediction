# Kế hoạch cải tổ cấu trúc thư mục dự án (MLOps Production-Ready)

Chào bạn, dưới góc độ là một Senior AI Engineer có 10 năm kinh nghiệm thiết kế hệ thống sản phẩm, mình hoàn toàn hiểu tại sao các Senior khác lại chê cây thư mục hiện tại của bạn. 

---

## 🔍 Tại sao cấu trúc hiện tại bị chê?
1. **Lẫn lộn giữa Package và Script**: Thư mục `src/` hiện tại vừa chứa logic cốt lõi (`data_loader.py`, `features.py`), vừa chứa các script điều khiển chạy (`predict_runner/main.py`), lại chứa cả ứng dụng web (`web/`). Trong MLOps chuẩn, `src/` chỉ nên chứa các module thư viện tái sử dụng được, còn các script chạy (huấn luyện, tuning, dọn dẹp) phải nằm riêng ở thư mục `scripts/` hoặc `bin/`.
2. **Rác ở Root Folder**: File như `clean_workspace.py` và `Hybrid_Model_Tasklist.md` nằm ở ngoài root tạo cảm giác lộn xộn. Root folder của một dự án chuyên nghiệp chỉ nên chứa README, LICENSE, requirements.txt và các thư mục cha.
3. **Thư mục `results/` quá hỗn tạp**: Thư mục này vừa chứa biểu đồ kết quả (.png), vừa chứa file ghi log dự đoán lịch sử (`predictions_history.txt`). Logs và hình ảnh trực quan hóa báo cáo nên được tách biệt rõ ràng (`logs/` và `reports/figures/`).
4. **Thiếu tính module hóa cấu hình**: File tham số tốt nhất `best_transformer_params.json` đang nằm chung trong thư mục lưu model (`models/`). Đúng ra nó phải thuộc thư mục cấu hình `config/`.

---

## 🛠️ Cấu trúc thư mục mới (Đề xuất)

```text
Stock-Opening-Price-Prediction/
├── config/                         # Cấu hình hệ thống và siêu tham số
│   └── best_transformer_params.json  # Di chuyển từ models/ sang config/
├── data/                           # Dữ liệu dự án
│   ├── raw/                        # File CSV dữ liệu thô (VNM_prices.csv,...)
│   └── processed/                  # Database SQLite (.db) và file cache cảm xúc (.csv)
├── docs/                           # Tài liệu dự án
│   ├── explain/                    # Hướng dẫn chi tiết (system_overview.md)
│   ├── PROJECT_HISTORY.md          # Nhật ký lịch sử
│   └── PROJECT_NOTES.md            # Báo cáo ghi chú kỹ thuật
├── logs/                           # Ghi log vận hành và lịch sử dự đoán
│   └── predictions_history.txt     # Di chuyển từ results/ sang logs/
├── models/                         # Lưu trữ file trọng số mô hình (.keras, .joblib)
├── notebooks/                      # Các file Jupyter Notebook tương tác (01_EDA.ipynb,...)
├── reports/                        # Báo cáo trực quan kết quả
│   └── figures/                    # Lưu các biểu đồ (.png) di chuyển từ results/
├── scripts/                        # Các script chạy độc lập (Tuning, Training, Utility)
│   ├── run_tuning.py               # (Đổi tên từ tune_transformer.py)
│   ├── run_training.py             # (Đổi tên từ hybrid_main.py)
│   └── clean_workspace.py          # Di chuyển từ root sang scripts/
├── src/                            # Lõi mã nguồn chính (Chỉ chứa logic cốt lõi)
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── news_sentiment.py
│   └── ai_models.py
├── web/                            # Web Application
│   ├── backend/                    # FastAPI Backend (api.py, db.py)
│   └── frontend/                   # HTML/CSS/JS Frontend
├── requirements.txt
├── README.md
├── LICENSE
└── Tasklist.md                     # Đổi tên từ Hybrid_Model_Tasklist.md
```

---

## 🚀 Các bước thực hiện di chuyển an toàn

Chúng ta sẽ tạo các thư mục mới, di chuyển các file và cập nhật toàn bộ đường dẫn imports cũng như đường dẫn file (File paths) trong các script Python để hệ thống không bị lỗi import:

### 1. Tạo thư mục mới & Di chuyển file
- Tạo các thư mục: `config/`, `logs/`, `reports/figures/`, `scripts/`, `data/raw/`, `data/processed/`.
- Di chuyển các file `.png` từ `results/` sang `reports/figures/`.
- Di chuyển `results/predictions_history.txt` sang `logs/predictions_history.txt`.
- Xóa bỏ thư mục `results/` sau khi đã chuyển hết.
- Di chuyển `models/best_transformer_params.json` sang `config/best_transformer_params.json`.
- Di chuyển các script chạy từ `src/predict_runner/` sang `scripts/` và đổi tên như cấu trúc trên. Xóa thư mục `predict_runner/`.
- Di chuyển `clean_workspace.py` từ root sang `scripts/clean_workspace.py`.
- Di chuyển các file giá thô vào `data/raw/`, các file SQLite và cache cảm xúc vào `data/processed/`.

### 2. Cập nhật mã nguồn (Sửa Imports & Paths)
- **`src/data_loader.py`**: Cập nhật đường dẫn đọc CSV thô từ `data/` sang `data/raw/`.
- **`src/web/backend/db.py` và `api.py`**: Cập nhật đường dẫn file SQLite `.db` và cache sang `data/processed/`.
- **`scripts/run_training.py`** và **`scripts/run_tuning.py`**:
  - Cập nhật imports từ `src.data_loader` thành imports chuẩn.
  - Cập nhật đường dẫn ghi kết quả biểu đồ `.png` sang `reports/figures/`.
  - Cập nhật đường dẫn ghi nhật ký dự báo sang `logs/predictions_history.txt`.
  - Cập nhật đường dẫn đọc `best_transformer_params.json` từ `config/`.
- **`notebooks/*.ipynb`**: Cập nhật các đường dẫn gọi file hoặc ghi ảnh trong các notebook tương thích với cấu trúc mới.

### 3. Kiểm thử & Xác thực
- Chạy thử `scripts/run_tuning.py` để xác thực cơ chế tìm kiếm siêu tham số và đọc file `config/`.
- Chạy thử `scripts/run_training.py` để xác thực quá trình huấn luyện và lưu biểu đồ vào `reports/figures/`.
- Khởi động thử backend FastAPI để đảm bảo việc đọc ghi database SQLite trong `data/processed/` không bị gián đoạn.

---

## Ý kiến của bạn?
Hãy xác nhận xem bạn có đồng ý với cấu trúc MLOps chuyên nghiệp này không để chúng ta bắt đầu thực thi việc sắp xếp lại cây thư mục.
