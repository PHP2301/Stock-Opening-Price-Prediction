from src.data_loader import load_stock_data
from src.models import StockPredictor
import matplotlib.pyplot as plt

def main():
    print("🚀 [HỆ THỐNG] Đang khởi động Pipeline dự báo giá mở cửa...")
    
    # 1. Nạp dữ liệu và tự động tính toán chỉ báo (đã tích hợp ở Ngày 3)
    file_path = 'data/VNM_prices.csv'
    df = load_stock_data(file_path)
    print(f"✅ [HỆ THỐNG] Đã chuẩn bị {len(df)} dòng dữ liệu sạch.")

    # 2. Khởi tạo bộ dự báo
    predictor = StockPredictor(df)

    # 3. Chạy thần tốc 2 mô hình cùng lúc
    lr_model, lr_results = predictor.train_linear_regression()
    xgb_model, xgb_results = predictor.train_xgboost()

    # 4. Trực quan hóa so sánh kết quả (Báo cáo trực quan)
    plt.figure(figsize=(14, 7))
    plt.plot(predictor.y_test.values[:100], label="Giá thực tế (Actual)", color='black', linewidth=2)
    plt.plot(lr_results["Predictions"][:100], label="Dự báo Linear Regression", linestyle='--', color='blue')
    plt.plot(xgb_results["Predictions"][:100], label="Dự báo XGBoost", linestyle='-.', color='red')
    
    plt.title("SO SÁNH HIỆU SUẤT DỰ BÁO GIÁ MỞ CỬA VINAMILK (100 phiên cuối)")
    plt.xlabel("Phiên giao dịch")
    plt.ylabel("Giá")
    plt.legend()
    
    # Lưu kết quả vào thư mục results
    plt.savefig('results/model_comparison.png')
    print("📊 [HỆ THỐNG] Đã xuất biểu đồ so sánh vào thư mục 'results/model_comparison.png'")
    plt.show()

if __name__ == "__main__":
    main()