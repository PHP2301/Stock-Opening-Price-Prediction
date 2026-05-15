# 📊 Stock-Opening-Price-Prediction Project

Nghiên cứu và ứng dụng các mô hình học máy (Linear Regression, LSTM, Random Forest) để dự báo giá mở cửa của các mã chứng khoán (VNM, GOOGL, META).

## 🚀 Các mô hình đã triển khai

1. **Linear Regression**: Mô hình tuyến tính cơ bản.
2. **Random Forest**: Mô hình dựa trên cây quyết định, xử lý tốt dữ liệu phi tuyến.
3. **LSTM (Long Short-Term Memory)**: Mạng nơ-ron hồi quy (RNN) chuyên dụng cho dữ liệu chuỗi thời gian.

---

## 📂 Cấu trúc thư mục

```
Stock-Opening-Price-Prediction/
├── data/
│   ├── GOOGL_prices.csv
│   ├── META_prices.csv
│   └── VNM_prices.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb          # Phân tích dữ liệu khám phá
│   ├── 02_FeatureEngineering.ipynb
│   ├── 03_LinearRegression.ipynb
│   ├── 04_RandomForest.ipynb
│   └── 05_LSTM.ipynb         # Mô hình Deep Learning
│
├── src/
│   ├── data_loader.py        # Module nạp và xử lý dữ liệu
│   ├── preprocessing.py      # Module chuẩn hóa và chia tách dữ liệu
│   └── model.py              # Module định nghĩa kiến trúc mô hình
│
└── requirements.txt          # Danh sách thư viện cần thiết
```

---

## 🛠️ Cài đặt và Chạy

### 1. Cài đặt môi trường

Đảm bảo bạn đã cài đặt Python 3.8+. Sau đó, tạo và kích hoạt môi trường ảo (khuyên dùng):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 2. Chạy Notebooks

Để chạy các bước phân tích và huấn luyện, bạn có thể sử dụng VS Code hoặc Jupyter Notebook.

**Ví dụ:** Mở Notebook số 01:

```bash
# Mở VS Code và chọn mở thư mục project
# Sau đó mở file 01_EDA.ipynb
```

**Hoặc chạy trực tiếp từ dòng lệnh với Jupyter:**

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 🎯 Các bước chính trong dự án

1. **EDA (Exploratory Data Analysis)**: Phân tích xu hướng giá và khối lượng giao dịch.
2. **Feature Engineering**: Tạo các biến đặc trưng (Lag features, Moving Averages).
3. **Model Training**: Huấn luyện 3 mô hình khác nhau.
4. **Evaluation**: So sánh hiệu năng của các mô hình dựa trên RMSE, MAE và R² Score.
