@echo off
:: Cấu hình UTF-8 để hiển thị tiếng Việt không bị lỗi font trên Windows Command Prompt
chcp 65001 > nul
title 🚀 RUN FULL STOCK PREDICTION PIPELINE

echo =====================================================================
echo    🚀 HỆ THỐNG DỰ BÁO GIÁ MỞ CỬA - PIPELINE CHẠY TOÀN BỘ HỆ THỐNG
echo =====================================================================
echo [*] Đang chuyển đến thư mục dự án...
cd /d "%~dp0.."

echo [*] Đang kích hoạt môi trường ảo Python (.venv)...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Không tìm thấy môi trường ảo .venv! Vui lòng cài đặt trước.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo    [TÙY CHỌN] BƯỚC 0: TỐI ƯU HÓA SIÊU THAM SỐ (HYPERPARAMETER TUNING)
echo =====================================================================
echo [*] Mặc định bước này bị bỏ qua vì chạy Optuna mất rất nhiều thời gian (30-40 phút/mã).
echo [*] Nếu bạn muốn chạy lại Tuning, hãy mở file scripts/run_full_pipeline.bat 
echo     và xóa các dấu "::" ở dòng chạy lệnh run_tuning.py.
echo.
:: echo [*] Đang chạy run_tuning.py...
:: python scripts/run_tuning.py
:: if %ERRORLEVEL% NEQ 0 (
::     echo [WARNING] Có lỗi xảy ra trong quá trình tuning. Tiếp tục...
:: )

echo.
echo =====================================================================
echo    [BƯỚC 1/4] HUẤN LUYỆN MÔ HÌNH (TRAINING) CHO TẤT CẢ CÁC MÃ
echo =====================================================================
echo [*] Đang chạy run_training_transformer.py...
python scripts/run_training_transformer.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Có lỗi xảy ra trong quá trình huấn luyện, tiếp tục các bước sau...
)

echo.
echo =====================================================================
echo    [BƯỚC 2/4] CHẠY BACKTEST TĨNH (STATIC BACKTEST) OUT-OF-SAMPLE
echo =====================================================================
echo [*] Đang chạy run_backtest.py...
python scripts/run_backtest.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Có lỗi xảy ra trong quá trình Backtest tĩnh, tiếp tục...
)

echo.
echo =====================================================================
echo    [BƯỚC 3/4] CHẠY ROLLING WALK-FORWARD BACKTEST (HỌC CUỘN)
echo =====================================================================
echo [*] Đang chạy run_walk_forward_backtest.py...
python scripts/run_walk_forward_backtest.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Có lỗi xảy ra trong quá trình Walk-Forward Backtest...
)

echo.
echo =====================================================================
echo    [BƯỚC 4/4] CHẠY DỰ BÁO THỰC TẾ HÀNG NGÀY KÈM MULTI-AGENT
echo =====================================================================
echo [*] Đang chạy predict.py cho toàn bộ các mã...
python scripts/predict.py ALL
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Lỗi khi chạy dự báo cuối cùng!
)

echo.
echo =====================================================================
echo    🏆 PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG!
echo    - Lịch sử dự báo hàng ngày: logs/predict_predictions_history.txt
echo    - Lịch sử huấn luyện mô hình: logs/train_predictions_history.txt
echo    - Biểu đồ equity curves: reports/figures/
echo =====================================================================
echo.
pause
