# Kế hoạch Nâng cấp Toàn diện Bộ trích xuất Đặc trưng Transformer

Để tối đa hóa chất lượng biểu diễn ẩn (embeddings) từ Transformer trước khi chuyển giao cho XGBoost, chúng ta sẽ thực hiện 3 cải tiến cấu trúc cốt lõi trực tiếp trên Transformer:

---

## Các thành phần cải tiến chi tiết:

### 1. Thêm Cổng Chọn Đặc trưng Đầu vào: Gated Linear Unit (GLU)
* **Giải pháp**: Ở đầu vào Transformer, sau lớp Conv1D, chúng ta thêm một lớp cổng GLU (`Gated Linear Unit`). 
* **Tác dụng**: Cổng GLU sẽ tự động học cách đóng/mở để lọc nhiễu, chỉ cho phép các chỉ báo kỹ thuật có tín hiệu mạnh nhất đi vào khối Self-Attention, trực tiếp giải quyết vấn đề nhiễu của dữ liệu tài chính.

### 2. Thay thế `Flatten` bằng `Bidirectional GRU`
* **Giải pháp**: Loại bỏ lớp `Flatten()` làm vỡ trình tự thời gian. Thay bằng một lớp **GRU hai chiều (Bidirectional GRU)** để nén chuỗi Attention 45 ngày thành một vector tóm tắt 64 chiều.
* **Tác dụng**: Giữ nguyên tính liên tục của chuỗi thời gian, bảo toàn các mẫu hình nến và giảm đột biến số lượng tham số từ 2880 chiều xuống còn 64 chiều để tránh overfit.

### 3. Áp dụng Học đa nhiệm: Thêm mục tiêu phụ (Auxiliary Target)
* **Giải pháp**: 
  * Tải và tính toán thêm mục tiêu phụ: Biên độ biến động ngày hôm sau (`target_spread = (high - low) / close`).
  * Nâng cấp Transformer thành mô hình 2 đầu ra (Multi-task output). Lớp Dense(32) sẽ là lớp cổ chai chia sẻ (Shared Bottleneck) phục vụ cả hai đầu ra: dự báo tỷ suất mở cửa (trọng số loss 1.0) và dự báo biên độ (trọng số loss 0.2).
* **Tác dụng**: Ép đặc trưng ẩn 32 chiều phải học cả xu thế giá lẫn mức độ biến động, tăng chất lượng thông tin của Embedding lên mức cao nhất.

---

## Proposed Changes

### [ML Pipeline & Core Code]

#### [MODIFY] [data_loader.py](file:///c:/Users/ACER/Documents/Stock-Opening-Price-Prediction/src/data_loader.py)
* Tính toán thêm cột `target_spread` cho nhãn phụ.

#### [MODIFY] [features.py](file:///c:/Users/ACER/Documents/Stock-Opening-Price-Prediction/src/features.py)
* Nâng cấp `DataTransformer` để nhận diện, scale và chia cửa sổ trượt đồng thời cho cả `target_return` và `target_spread`.

#### [MODIFY] [ai_models.py](file:///c:/Users/ACER/Documents/Stock-Opening-Price-Prediction/src/ai_models.py)
* Tích hợp GLU layer.
* Thay `Flatten` bằng `Bidirectional(GRU)`.
* Cấu hình Multi-task output với 2 hàm loss Huber.

#### [MODIFY] [run_training.py](file:///c:/Users/ACER/Documents/Stock-Opening-Price-Prediction/scripts/run_training.py)
* Cập nhật luồng huấn luyện chéo K-Fold và huấn luyện Transformer chính thích ứng với 2 đầu ra dữ liệu.

---

## Ý kiến của bạn?
Hãy xác nhận **"Đồng ý"** để tôi tiến hành nâng cấp sâu toàn bộ kiến trúc mạng Transformer này cho bạn!
