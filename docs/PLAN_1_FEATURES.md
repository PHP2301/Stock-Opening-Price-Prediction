# BẢN KẾ HOẠCH 1: TÍCH HỢP 8 FEATURES MỚI (PHASE 1)

## Mục tiêu
Nâng cấp input của mô hình từ 34 đặc trưng lên 42 đặc trưng bằng cách thêm các nhóm chỉ báo MFI, Cổ tức và Proxy giao dịch khối ngoại/tự doanh. Sau đó retrain lại toàn bộ mô hình để tương thích với bộ data mới.

## Danh sách 8 Features
1. `mfi_14` (Money Flow Index)
2. `dividend_flag` (Tín hiệu ngày ex-right)
3. `days_to_dividend` (Ngày đến kỳ cổ tức tiếp theo)
4. `days_after_dividend` (Ngày kể từ kỳ cổ tức trước)
5. `foreign_net_buy_proxy` (Proxy khối ngoại dựa trên volume & giá)
6. `foreign_net_buy_5d` (Proxy ngoại trung bình 5 ngày)
7. `foreign_net_buy_20d` (Proxy ngoại trung bình 20 ngày)
8. `self_net_buy_proxy` (Proxy tự doanh dựa trên MFI divergence)

## Các bước thực hiện chi tiết

### Bước 1: [x] Tạo module xử lý Cổ tức (`src/dividend_fetcher.py`)
- [x] Viết hàm lấy lịch sử ngày chốt quyền (ex-right dates) từ Yahoo Finance cho `GOOGL`, `META`.
- [x] Viết hàm dùng `vnstock` (`Company.events()`) lấy ngày chốt quyền chia cổ tức bằng tiền mặt cho `VNM.VN`.
- [x] Hợp nhất và trả về một list các ngày chia cổ tức.

### Bước 2: [x] Cập nhật luồng xử lý Dữ liệu (`src/data_loader.py`)
- [x] Gọi module `dividend_fetcher` vào hàm `fetch_and_prepare_data`.
- [x] Gắn thêm cột `dividend_flag` vào DataFrame dựa trên ngày giao dịch.
- [x] Tính toán 2 cột đếm ngày: `days_to_dividend` (đếm lùi) và `days_after_dividend` (đếm tới).

### Bước 3: [x] Cập nhật Feature Engineering (`src/features.py`)
- [x] Trong class `DataTransformer`, mở rộng `feature_cols` lên thành 42 cột.
- [x] Thêm logic tính toán `mfi_14` bằng `pandas_ta`.
- [x] Tính `foreign_net_buy_proxy` thông qua Z-score của Volume kết hợp với Close Location Value (CLV).
- [x] Tạo các cột rolling 5d, 20d cho foreign proxy.
- [x] Tính `self_net_buy_proxy` thông qua sự phân kỳ của MFI và biến động giá.
- [x] Update các input layer của mô hình lai (Concatenate size = 64 thay vì 56).

### Bước 4: [x] Retrain toàn bộ mô hình (`scripts/run_training.py`)
- [x] Xóa các model cũ đã lưu.
- [x] Chạy lại huấn luyện cho `VNM.VN`, `GOOGL`, `META`.
- [x] Ghi nhận MAE, RMSE mới và so sánh với phiên bản 34 features.

## Thời gian dự kiến: 1 ngày làm việc
