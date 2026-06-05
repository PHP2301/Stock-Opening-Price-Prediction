# Kế hoạch Triển khai Chính thức: Nâng cấp Toàn diện Mô hình Dự báo giá mở cửa

Tài liệu này là đặc tả kỹ thuật chi tiết nhất đã được thống nhất để triển khai bộ 34 đặc trưng High-Alpha, cấu trúc Transformer phân nhánh, cơ chế Time-Decay Self-Attention, hàm Loss trọng số không xác định (Kendall 2018), và quy trình kiểm thử Walk-Forward Validation nghiêm ngặt.

---

## 📋 Các thành phần nâng cấp chi tiết

### PHA 1: Khử Rò Rỉ Dữ Liệu & Đồng Bộ Múi Giờ
1. **Đồng bộ múi giờ NYSE vs HOSE (Cho mã Việt Nam `VNM.VN`)**:
   * Toàn bộ dữ liệu vĩ mô đóng cửa phiên Mỹ ngày $T$ (`vix`, `bond_yield_10y`, `dollar_index_change`, tỷ suất S&P 500 qua đêm) sẽ được **dịch trễ 1 ngày (shift 1)**.
   * Đảm bảo tại thời điểm chạy mô hình sau Close ngày $T$, chúng ta chỉ dùng dữ liệu phiên Mỹ đã đóng cửa lúc 04:00 ICT sáng ngày $T$ (tức phiên Mỹ $T-1$).
   * Đối với mã Mỹ (`GOOGL`/`META`), chạy cùng múi giờ nên giữ nguyên, không shift trễ.
2. **Khử rò rỉ mẫu số giá đóng cửa**:
   * Đặc trưng được tính toán **sau Close T** để đặt lệnh khớp ở **Open T+1**.
   * Mọi đặc trưng sử dụng Close(t) làm mẫu số đều hợp lệ vì Close(t) đã xác định tại thời điểm tính toán sau Close phiên ngày T.
   * Loại bỏ hoàn toàn `close_lag1_ratio = Close(t-1) / Close(t)` và `open_lag1_ratio = Open(t-1) / Close(t)` để tránh gây nhầm lẫn phi tĩnh.
   * Thay thế bằng tỷ suất sinh lời stationary:
     * `return_1d = Close(t-1) / Close(t-2) - 1`
     * `return_2d = Close(t-2) / Close(t-3) - 1`
     * `return_3d = Close(t-3) / Close(t-4) - 1`
     * `open_return = Open(t) / Open(t-1) - 1` (Gap qua đêm thực sự).

---

## PHA 2: Thiết kế Hệ thống 34 Đặc trưng & Phân nhánh Đầu vào

34 đặc trưng cơ bản (28 phi thời gian + 6 lịch) được phân chia vào 3 nhánh đầu vào độc lập:

#### Nhánh 1 — Giá & Động lượng (12 Đặc trưng)
1. `gap_open`: `Open(t) / Close(t-1) - 1` (Gap mở cửa qua đêm)
2. `open_return`: `Open(t) / Open(t-1) - 1` (Thay thế cho close_lag5_ratio bị trùng)
3. `buying_pressure`: `(Close(t) - Low(t)) / (High(t) - Low(t) + 1e-9)` (Áp lực mua cuối phiên ∈ [0, 1])
4. `shadow_ratio`: `(High(t) - Close(t)) / (Close(t) - Low(t) + 1e-9)` (Bóng nến trên/dưới)
5. `intraday_range`: `(High(t) - Low(t)) / Close(t)` (Biên độ dao động trong ngày)
6. `return_1d`: `Close(t-1) / Close(t-2) - 1`
7. `return_2d`: `Close(t-2) / Close(t-3) - 1`
8. `return_3d`: `Close(t-3) / Close(t-4) - 1`
9. `mom_5d`: `Close(t) / Close(t-5) - 1` (Xu hướng 1 tuần)
10. `mom_10d`: `Close(t) / Close(t-10) - 1` (Xu hướng 2 tuần)
11. `mom_20d`: `Close(t) / Close(t-20) - 1` (Xu hướng 1 tháng)
12. `dist_ma50`: `Close(t) / Mean(Close[t-49:t]) - 1` (MA50 tính inclusive Close(t))

#### Nhánh 2 — Khối lượng & Biến động (6 Đặc trưng)
13. `volume_change`: `Volume(t) / Volume(t-1) - 1` (Thay đổi khối lượng 1 ngày)
14. `volume_sma_ratio`: `Volume(t) / Mean(Volume[t-19:t])` (Khối lượng so với trung bình 20 ngày)
15. `volume_zscore`: `(Volume(t) - Mean_Vol_20) / (Std_Vol_20 + 1e-9)` (Đột biến dòng tiền)
16. `ad_line_ratio`: `((Close - Low) - (High - Close)) / (High - Low + 1e-9)` (Tích lũy/phân phối chuẩn hóa về [-1, 1], không nhân hay chia Volume nhằm tránh triệt tiêu toán học)
17. `obv_zscore`: Đặc trưng dòng tiền chuẩn hóa (Z-score 20 phiên của hiệu số OBV 5 phiên):
    $$\text{delta\_obv} = \text{OBV}(t) - \text{OBV}(t-5)$$
    $$\text{std\_20} = \text{rolling\_std}(\text{delta\_obv}, \text{window}=20)$$
    $$\text{obv\_zscore} = \frac{\text{delta\_obv}}{\text{std\_20} + 1e-9}$$
18. `vol_ratio`: `volatility_5d / (volatility_60d + 1e-9)` (Bùng nổ biến động)

#### Nhánh 3 — Kỹ thuật, Vĩ mô & Lịch (16 Đặc trưng)
19. `rsi_14`: Chỉ số RSI 14 ngày
20. `macd_ratio`: `MACD / (Signal + 1e-9)`
21. `bb_position`: `(Close - LowerBand) / (UpperBand - LowerBand + 1e-9)` (Vị thế trong dải Bollinger ∈ [0, 1])
22. `adx_14`: Average Directional Index 14 ngày
23. `stoch_k`: Stochastic %K
24. `efficiency_ratio`: `abs(Close(t) - Close(t-10)) / (Sum(abs(daily_changes_10d)) + 1e-9)` (Tránh chia cho 0 khi đi ngang)
25. `vix_lag1`: VIX shift 1 (đối với VN) hoặc VIX ngày hiện tại (đối với Mỹ)
26. `bond_yield_lag1`: US 10Y Treasury yield dịch trễ 1 ngày
27. `usdvnd_change`: `USDVND(t) / USDVND(t-1) - 1`
28. `vnindex_return_lag1`: Tỷ suất VNINDEX (VN) hoặc NASDAQ (Mỹ) dịch trễ 1 ngày
29. `day_of_week_sin`: `sin(2π * DOW / 5)` (Tránh discontinuity)
30. `day_of_week_cos`: `cos(2π * DOW / 5)`
31. `month_sin`: `sin(2π * Month / 12)`
32. `month_cos`: `cos(2π * Month / 12)`
33. `is_quarter_end`: `1` nếu thuộc 3 ngày giao dịch cuối quý, ngược lại là `0`
34. `days_before_tet`: `min(số phiên giao dịch còn lại đến Tết, 30)` (Cap tại 30 phiên, reset về 30 ngay sau Tết, chỉ áp dụng cho VN)

---

### PHA 3: Kiến trúc Transformer Phân Nhánh & Học đa nhiệm
1. **Time-Decay Attention**:
   * Tính ma trận khoảng cách giữa các phiên $|i - j|$ trong chuỗi 45 ngày.
   * `Penalty = -exp(log_gamma) * |i - j|` với `log_gamma` là một trainable weight riêng cho từng head attention của mỗi nhánh.
2. **Cơ chế Phân nhánh**:
   * Đầu vào `(N, 45, 34)` tách thành 3 nhánh tương ứng qua Conv1D + TimeDecayAttention để tạo ra:
     * `latent_price` (16 chiều)
     * `latent_volume` (8 chiều)
     * `latent_tech` (8 chiều)
   * `Concatenate` thành vector **32 chiều** làm Bottleneck representation.
3. **Định nghĩa Học đa nhiệm & Trọng số không xác định (Kendall 2018)**:
   * **Task 1 (Chính)**: Dự báo liên tục tỷ suất sinh lời mở cửa ngày mai `target_return = Open(T+1) / Close(T) - 1`.
   * **Task 2 (Phụ)**: Dự báo chênh lệch dao động giá ngày mai `target_spread = (High(T+1) - Low(T+1)) / Close(T)`.
   * Chuẩn hóa nhãn target trước khi tính Loss:
     $$\text{return\_normalized} = \frac{\text{Return} - \mu_{\text{return}}}{\sigma_{\text{return}}}$$
     $$\text{spread\_normalized} = \frac{\text{Spread} - \mu_{\text{spread}}}{\sigma_{\text{spread}}}$$
   * **Uncertainty Weighting (Kendall 2018)**: Sử dụng các tham số học tập $\sigma_1, \sigma_2$ để tự động cân bằng tỷ lệ gradient:
     $$\text{Loss}_{\text{total}} = \frac{1}{2\sigma_1^2} \text{HuberLoss}(\text{return\_norm}) + \text{HuberLoss}(\text{spread\_norm}) + \log(\sigma_1) + \log(\sigma_2)$$

---

### PHA 4: Walk-Forward Validation & Chi Phí Thực Tế
1. **Walk-Forward với Purge Gap 45 PHIÊN GIAO DỊCH (Trading Days)**:
   * Để triệt tiêu rò rỉ dữ liệu qua chuỗi overlap do cửa sổ trượt 45 phiên gây ra, Purge Gap được xác định chuẩn xác theo **số phiên giao dịch thực tế** trên thị trường (chỉ mục dòng của DataFrame lịch sử đã lọc ngày nghỉ/lễ) chứ không cộng theo ngày lịch.
   * Công thức chia Folds sử dụng chỉ mục DataFrame:
     * **Fold 1**: Train `[Idx_start, Idx_train_end]` | Purge Gap (Bỏ) `[Idx_train_end + 1, Idx_train_end + 45]` | Test `[Idx_train_end + 46, Idx_test_end]`
2. **Chi phí giao dịch & Dynamic Slippage**:
   * Chi phí cơ sở:
     * **VNM.VN (Bluechip)**: Phí sàn 0.15% + Slippage 0.10% + Market impact 0.05% = 0.30% một chiều (0.60% khứ hồi).
     * **Mã VN thanh khoản thấp**: Phí sàn 0.15% + Slippage 0.40%–0.80% + Market impact 0.10% = 0.65%–1.05% một chiều.
     * **Mã Mỹ (GOOGL/META)**: Phí sàn 0.05% + Slippage 0.02% = 0.07% một chiều (0.14% khứ hồi).
   * **Dynamic Slippage**: Trong backtest, nếu `volume_zscore < -1` (khối lượng thấp bất thường), nhân hệ số slippage lên **1.5× – 2.0×**.

---

## Danh sách tệp tin thay đổi:

*   **`src/features.py`**: Tích hợp 34 đặc trưng cơ bản mới và xử lý chuẩn hóa chống rò rỉ dữ liệu, epsilon guard.
*   **`src/data_loader.py`**: Xử lý dịch trễ vĩ mô Mỹ cho mã VN và tính toán Tết Bounded, Quarter End.
*   **`src/ai_models.py`**: Xây dựng lớp `TimeDecayAttention` và kiến trúc Transformer phân 3 nhánh đầu vào.
*   **`src/web/backend/api.py`**: Điều chỉnh tính toán đặc trưng thời gian thực cho 34 đặc trưng đầu vào mới.
*   **`scripts/run_training.py`** & **`scripts/run_backtest.py`**: Tích hợp chia tập Walk-forward với Purge Gap 45 ngày và Backtest tính dynamic slippage.
