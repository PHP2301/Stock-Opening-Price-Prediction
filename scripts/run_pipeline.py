import sys
import os
import subprocess

def main():
    # Lấy ticker từ tham số dòng lệnh hoặc biến môi trường
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else os.getenv("TICKER", "").upper()
    
    print("\n=======================================================")
    print(f"🚀 KHỞI CHẠY PIPELINE HUẤN LUYỆN CHO: {ticker if ticker else 'TẤT CẢ CÁC MÃ'}")
    print("=======================================================\n")
    
    # Bước 1: Huấn luyện mô hình production
    print("=== [BƯỚC 1/3] Huấn luyện mô hình production ===")
    cmd_train = [sys.executable, "scripts/run_training_transformer.py"]
    if ticker:
        cmd_train.append(ticker)
    subprocess.run(cmd_train, check=True)
    
    # Bước 2: Chạy Rolling Walk-Forward Backtest
    print("\n=== [BƯỚC 2/3] Chạy Rolling Walk-Forward Backtest ===")
    cmd_wf = [sys.executable, "scripts/run_walk_forward_backtest.py"]
    if ticker:
        cmd_wf.append(ticker)
    subprocess.run(cmd_wf, check=True)
    
    # Bước 3: Chạy dự báo thực tế hàng ngày (để chạy thử nghiệm agents)
    print("\n=== [BƯỚC 3/3] Chạy thử dự báo hàng ngày với AI Agents ===")
    cmd_pred = [sys.executable, "scripts/predict.py"]
    cmd_pred.append(ticker if ticker else "ALL")
    subprocess.run(cmd_pred, check=True)
    
    print("\n=======================================================")
    print("🏆 PIPELINE HUẤN LUYỆN VÀ DỰ BÁO ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
