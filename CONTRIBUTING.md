# Hướng dẫn Đóng góp (Contributing Guidelines)

Chào mừng bạn đã đến với dự án dự báo giá mở cửa bằng mô hình lai! Chúng tôi rất trân trọng mọi đóng góp từ cộng đồng nhằm tối ưu hóa các chiến lược giao dịch định lượng và các kiến trúc học máy nâng cao.

## 🚀 Quy trình Đóng góp

1. **Fork repository** này về tài khoản cá nhân của bạn.
2. **Tạo nhánh mới (Branch)** cho tính năng hoặc bản sửa lỗi:
   ```bash
   git checkout -b feature/awesome-feature
   ```
3. **Thực hiện các thay đổi** và chạy kiểm thử nội bộ để đảm bảo chất lượng:
   ```bash
   python -m unittest discover -s tests
   ```
4. **Cam kết các thay đổi (Commit)** với thông điệp rõ ràng:
   ```bash
   git commit -m "feat: thêm đặc trưng vi cấu trúc thị trường mới"
   ```
5. **Đẩy lên nhánh của bạn (Push)**:
   ```bash
   git push origin feature/awesome-feature
   ```
6. **Mở Pull Request (PR)** giải thích chi tiết mục đích và kết quả thực nghiệm.

## 🛠️ Yêu cầu Mã nguồn & Tiêu chuẩn

- **Python Version**: Khuyến nghị sử dụng Python `3.10` hoặc `3.11`.
- **Định dạng Code**: Sử dụng chuẩn PEP 8. Bạn nên chạy format tự động (ví dụ `black` hoặc `ruff`) trước khi commit.
- **Tính nhất quán của Đặc trưng**: Bất kỳ sự thay đổi hoặc bổ sung đặc trưng nào trong `src/features.py` cũng cần đồng bộ tại `src/data_loader.py` và `src/web/backend/api.py`.
- **Kiểm thử**: Viết unit test cho các tính năng cốt lõi mới trong thư mục `tests/`.

Cảm ơn bạn đã đồng hành đóng góp xây dựng hệ thống dự báo tốt hơn!
