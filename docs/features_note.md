# Danh sách Chi tiết 66 Đặc trưng Hệ thống (Mới nhất - Bản chỉnh sửa 5)

Tài liệu này liệt kê chi tiết toàn bộ **66 đặc trưng** được sử dụng trong mô hình lai **Hybrid XGBoost** sau khi loại bỏ rò rỉ dữ liệu, trùng lặp chỉ báo, chuẩn hóa động lượng khối lượng và tích hợp đặc trưng thời gian (Calendar) có giới hạn chu kỳ.

---

## A. Nhóm đặc trưng cơ bản (34 Đặc trưng)

Được chia thành 3 nhánh tương thích với cấu trúc mạng Transformer phân nhánh:

### Nhánh 1 — Giá & Động lượng (12 Đặc trưng)

| STT | Tên đặc trưng | Loại | Ý nghĩa / Cách tính |
| :--- | :--- | :--- | :--- |
| 1 | `gap_open` | Cấu trúc vi mô | Tỷ suất gap mở cửa qua đêm: `Open(t) / Close(t-1) - 1` |
| 2 | `open_return` | Động lượng | Gap qua đêm thực sự: `Open(t) / Open(t-1) - 1` |
| 3 | `buying_pressure` | Cấu trúc vi mô | Áp lực mua cuối ngày: `(Close(t) - Low(t)) / (High(t) - Low(t) + 1e-9)` |
| 4 | `shadow_ratio` | Cấu trúc vi mô | Tỷ lệ râu nến trên/dưới: `(High(t) - Close(t)) / (Close(t) - Low(t) + 1e-9)` |
| 5 | `intraday_range` | Cấu trúc vi mô | Biên độ dao động trong phiên: `(High(t) - Low(t)) / Close(t)` |
| 6 | `return_1d` | Động lượng | Tỷ suất đóng cửa ngày hôm trước: `Close(t-1) / Close(t-2) - 1` |
| 7 | `return_2d` | Động lượng | Tỷ suất đóng cửa 2 ngày trước: `Close(t-2) / Close(t-3) - 1` |
| 8 | `return_3d` | Động lượng | Tỷ suất đóng cửa 3 ngày trước: `Close(t-3) / Close(t-4) - 1` |
| 9 | `mom_5d` | Động lượng | Tỷ suất sinh lời đóng cửa 5 ngày qua (tuần): `Close(t) / Close(t-5) - 1` |
| 10 | `mom_10d` | Động lượng | Tỷ suất sinh lời đóng cửa 10 ngày qua (2 tuần): `Close(t) / Close(t-10) - 1` |
| 11 | `mom_20d` | Động lượng | Tỷ suất sinh lời đóng cửa 20 ngày qua (tháng): `Close(t) / Close(t-20) - 1` |
| 12 | `dist_ma50` | Xu hướng | Khoảng cách giá đóng cửa tới MA50: `Close(t) / Mean(Close[t-49:t]) - 1` |

### Nhánh 2 — Khối lượng & Biến động (6 Đặc trưng)

| STT | Tên đặc trưng | Loại | Ý nghĩa / Cách tính |
| :--- | :--- | :--- | :--- |
| 13 | `volume_change` | Khối lượng | Tỷ lệ thay đổi khối lượng giao dịch so với hôm trước: `Volume(t) / Volume(t-1) - 1` |
| 14 | `volume_sma_ratio` | Khối lượng | Tỷ lệ khối lượng hôm nay so với trung bình 20 ngày: `Volume(t) / Mean(Volume[t-19:t])` |
| 15 | `volume_zscore` | Khối lượng | Độ đột biến dòng tiền: `(Volume(t) - Mean_Vol_20) / (Std_Vol_20 + 1e-9)` |
| 16 | `ad_line_ratio` | Dòng tiền | Vị trí giá tương đối tích lũy/phân phối chuẩn hóa: `((Close - Low) - (High - Close)) / (High - Low + 1e-9)` |
| 17 | `obv_zscore` | Dòng tiền | Z-score hiệu số OBV 5 ngày so với lịch sử 20 ngày: `delta_obv / (std_20 + 1e-9)` với `delta_obv = OBV(t) - OBV(t-5)` và `std_20 = rolling_std(delta_obv, 20)` |
| 18 | `vol_ratio` | Biến động | Tỷ lệ biến động ngắn/dài hạn: `volatility_5d / (volatility_60d + 1e-9)` |

### Nhánh 3 — Kỹ thuật, Vĩ mô & Lịch (16 Đặc trưng)

| STT | Tên đặc trưng | Loại | Ý nghĩa / Cách tính |
| :--- | :--- | :--- | :--- |
| 19 | `rsi_14` | Kỹ thuật | Chỉ số sức mạnh tương đối (RSI) 14 ngày |
| 20 | `macd_ratio` | Kỹ thuật | Tỷ lệ đường MACD / Tín hiệu MACD |
| 21 | `bb_position` | Kỹ thuật | Vị thế giá trong Bollinger Bands: `(Close - Lower) / (Upper - Lower + 1e-9)` |
| 22 | `adx_14` | Kỹ thuật | Chỉ số định hướng trung bình ADX 14 ngày |
| 23 | `stoch_k` | Kỹ thuật | Chỉ báo dao động Stochastic Oscillator đường %K |
| 24 | `efficiency_ratio` | Trạng thái | Hệ số hiệu quả xu hướng (Trending vs Ranging) kèm Epsilon Guard |
| 25 | `vix_lag1` | Vĩ mô | Chỉ số VIX dịch trễ 1 ngày (VN) hoặc VIX ngày hiện tại (Mỹ) |
| 26 | `bond_yield_lag1` | Vĩ mô | Lợi suất trái phiếu chính phủ Mỹ 10Y dịch trễ 1 ngày |
| 27 | `usdvnd_change` | Vĩ mô | Biến động của tỷ giá USD/VND (cho VN) |
| 28 | `vnindex_return_lag1` | Vĩ mô | Tỷ suất VNINDEX dịch trễ 1 ngày (VN) hoặc NASDAQ ngày hiện tại (Mỹ) |
| 29 | `day_of_week_sin` | Lịch | Dạng sóng Sin của thứ trong tuần `sin(2π * DOW / 5)` |
| 30 | `day_of_week_cos` | Lịch | Dạng sóng Cos của thứ trong tuần `cos(2π * DOW / 5)` |
| 31 | `month_sin` | Lịch | Dạng sóng Sin của tháng trong năm `sin(2π * Month / 12)` |
| 32 | `month_cos` | Lịch | Dạng sóng Cos của tháng trong năm `cos(2π * Month / 12)` |
| 33 | `is_quarter_end` | Lịch | Nhận diện ngày cuối quý (chốt NAV quỹ): `1` nếu đúng, `0` nếu sai |
| 34 | `days_before_tet` | Lịch | Chỉ số cận Tết Nguyên Đán giới hạn: `min(số phiên giao dịch còn lại đến Tết, 30)` |

---

## B. Nhóm đặc trưng ẩn - Latent Embedding (32 Đặc trưng)

Mô hình học sâu nén thông tin chuỗi 45 ngày lịch sử của các đặc trưng trên thành 3 nhóm biểu diễn ẩn độc lập trước khi Concatenate:
*   `latent_price` (16 chiều, từ Nhánh Giá)
*   `latent_volume` (8 chiều, từ Nhánh Khối lượng)
*   `latent_tech` (8 chiều, từ Nhánh Tech/Macro/Calendar)

| STT | Tên đặc trưng | Nhóm nguồn | Mô tả |
| :--- | :--- | :--- | :--- |
| 35 | `latent_0` | `latent_price` | Đặc trưng ẩn giá số 1 trích xuất từ 45 ngày lịch sử nến |
| ... | ... | ... | ... |
| 50 | `latent_15` | `latent_price` | Đặc trưng ẩn giá số 16 trích xuất từ 45 ngày lịch sử nến |
| 51 | `latent_16` | `latent_volume` | Đặc trưng ẩn dòng tiền số 1 trích xuất từ 45 ngày lịch sử khối lượng |
| ... | ... | ... | ... |
| 58 | `latent_23` | `latent_volume` | Đặc trưng ẩn dòng tiền số 8 trích xuất từ 45 ngày lịch sử khối lượng |
| 59 | `latent_24` | `latent_tech` | Đặc trưng ẩn kỹ thuật/vĩ mô số 1 trích xuất từ 45 ngày lịch sử vĩ mô/lịch |
| ... | ... | ... | ... |
| 66 | `latent_31` | `latent_tech` | Đặc trưng ẩn kỹ thuật/vĩ mô số 8 trích xuất từ 45 ngày lịch sử vĩ mô/lịch |
