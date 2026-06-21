# Báo cáo Kết quả Đối chứng: Sửa lỗi Portfolio Stop-Loss Lockout & Đánh giá Triển khai

Chúng ta đã hoàn thành việc triển khai cơ chế **Resume (Khôi phục giao dịch)** với **20 phiên Cooldown** cho cơ chế Portfolio Stop-loss trong engine dùng chung `src/backtest_engine.py`. Sau đó, chúng ta đã chạy lại toàn bộ các bài kiểm thử tĩnh (Static) và cuộn chiếu (Rolling Walk-Forward) cho hai mã Mỹ là **META** và **GOOGL**.

Kết quả thực tế đã làm sáng tỏ một sự thật quan trọng về hiệu năng của mô hình.

---

## 1. Bảng So Sánh Hiệu Suất Trước và Sau Khi Sửa Stop-Loss

Dưới đây là bảng so sánh chi tiết các chỉ số chính (Lợi nhuận, Sharpe, Drawdown, Số lệnh) của mô hình Hybrid trước và sau khi áp dụng cơ chế tự động mở lại giao dịch:

### META

| Phương pháp | Trạng thái Stop-Loss | Số lệnh (Trades) | Tỷ lệ thắng (Win Rate) | Lợi nhuận (Return) | Lợi nhuận B&H | Sharpe Ratio | Max Drawdown (MDD) |
|---|---|---|---|---|---|---|---|
| **Static (Cũ)** | Bị khóa vĩnh viễn | 46 | 69.57% | **+31.93%** | +89.24% | **0.78** | **-20.32%** |
| **Static (Mới)** | **Resume + 20d Cooldown** | 61 | 60.66% | **+11.68%** | +89.24% | **0.33** | **-32.55%** |
| **Rolling WF (Cũ)**| Bị khóa vĩnh viễn | 24 | 58.33% | **+58.59%** | +397.32% | **0.79** | **-20.99%** |
| **Rolling WF (Mới)**| **Resume + 20d Cooldown** | 90 | 52.22% | **-16.69%** | +397.32% | **-0.13**| **-40.69%** |

### GOOGL

| Phương pháp | Trạng thái Stop-Loss | Số lệnh (Trades) | Tỷ lệ thắng (Win Rate) | Lợi nhuận (Return) | Lợi nhuận B&H | Sharpe Ratio | Max Drawdown (MDD) |
|---|---|---|---|---|---|---|---|
| **Static (Cũ)** | Bị khóa vĩnh viễn | 24 | 54.17% | **-17.50%** | +193.78% | **-0.56** | **-20.03%** |
| **Static (Mới)** | **Resume + 20d Cooldown** | 34 | 55.88% | **-4.94%** | +193.78% | **-0.08** | **-20.03%** |
| **Rolling WF (Cũ)**| Bị khóa vĩnh viễn | 31 | 58.06% | **-5.44%** | +355.23% | **-0.13** | **-21.02%** |
| **Rolling WF (Mới)**| **Resume + 20d Cooldown** | 65 | 50.77% | **-0.01%** | +355.23% | **+0.08**| **-30.22%** |

### VNM.VN (Không đổi vì không kích hoạt dừng lỗ)
- **Static/Rolling**: Return **-2.69%** (B&H: -9.51%), Sharpe **-0.40**, MDD **-5.13%**, Số lệnh: 4.

---

## 2. Phân Tích & Phát Hiện Quan Trọng

> [!WARNING]
> ### Sự thật về mã META: Lỗi thiết kế stop-loss đã che giấu rủi ro lớn
> - **Trước đây**, META được coi là ứng viên triển khai (Deploy Candidate) triển vọng nhất với Sharpe ~0.78.
> - **Thực tế sau khi sửa**: Khi cho phép mở lại giao dịch sau dừng lỗ, kết quả của META sụt giảm mạnh (Rolling WF Return giảm từ **+58.59%** xuống **-16.69%**, Sharpe giảm xuống **-0.13**, và Max Drawdown tăng vọt lên **-40.69%**).
> - **Lý do**: Cơ chế khóa vĩnh viễn cũ đã vô tình hoạt động như một "tấm khiên" bảo vệ mô hình bằng cách ép nó giữ 100% tiền mặt ngay trước khi thị trường bước vào giai đoạn điều chỉnh/sụp đổ kéo dài ở nửa sau của chu kỳ test (2025/2026). Khi cho phép resume, mô hình tiếp tục dự đoán sai trong thị trường giá xuống, dẫn đến việc bị quét stop-loss lần thứ 2 và bào mòn vốn nặng nề.

> [!NOTE]
> ### Tác động đối với GOOGL: Cải thiện kỹ thuật nhưng chưa đủ alpha
> - Khi cho phép resume, GOOGL được tham gia trở lại vào Window 3 (giai đoạn phục hồi mạnh của cổ phiếu này). Hiệu suất Window 3 đạt **+13.68%** (Sharpe **0.94**, MDD **-14.21%**) thay vì bị giữ ở mức +0.00% như trước.
> - Điều này giúp Sharpe tổng thể của Rolling WF tăng từ **-0.13** lên **+0.08** (gần hòa vốn). Tuy nhiên, chiến lược vẫn thua xa chỉ số Buy & Hold (+355.23%) và tổng thể vẫn có alpha rất yếu.

---

## 3. Khuyến Nghị Quyết Định Triển Khai Cuối Cùng (Final Deployment Decision)

Sau khi sửa lỗi dừng lỗ lockout và có kết quả đối chứng khách quan, chúng ta tiến hành phân loại lại toàn bộ danh mục như sau:

| Ticker | Đánh giá trước đây | Phân loại mới (Đã sửa lỗi) | Quyết định triển khai |
|---|---|---|---|
| **META** | Deploy Candidate ( Sharpe ~0.78 ) | Rủi ro cao / Bị cưa chân bàn khi thị trường giảm ( Sharpe -0.13, MDD -40% ) | **🔴 KILL-SWITCH (Không triển khai)** |
| **GOOGL** | Kill-switch / Sharpe âm | Alpha quá yếu / Không đánh bại được B&H ( Sharpe +0.08 ) | **🔴 KILL-SWITCH (Không triển khai)** |
| **VNM.VN**| Kill-switch / Giao dịch quá ít | Alpha âm / Thiếu thanh khoản/tín hiệu ( Sharpe -0.40 ) | **🔴 KILL-SWITCH (Không triển khai)** |

### Kết luận:
Không có mã cổ phiếu nào vượt qua được bộ lọc an toàn để triển khai live. Việc dừng lỗi thiết kế lockout đã giúp bảo vệ vốn của nhà đầu tư bằng cách bộc lộ những điểm yếu thực tế của mô hình dự báo trong các chu kỳ downtrend/sideway.
