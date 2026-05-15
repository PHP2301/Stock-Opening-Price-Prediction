import pandas as pd

def load_stock_data(file_path):
    # Bước 1: Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Bước 2: Chuẩn hóa tên cột (VNM là 'time', Google/Meta là 'Date')
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'Date'})
    
    # Bước 3: Ép kiểu thời gian và sắp xếp
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Bước 4: Tạo biến mục tiêu (Giá mở cửa ngày mai)
    df['Next_Open'] = df['open'].shift(-1)
    
    return df


if __name__ == "__main__":
    # 1. Khởi tạo đường dẫn
    file_path = 'data/VNM_prices.csv'

    # 2. Gọi hàm xử lý
    df_processed = load_stock_data(file_path)

    # 3. Kiểm tra kết quả (Xem 5 dòng cuối cùng, vì dòng cuối bị mất số liệu Next_Open)
    print(df_processed.tail())