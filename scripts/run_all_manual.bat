@echo off
:: Cấu hình UTF-8 để hiển thị tiếng Việt không bị lỗi font trên cmd Windows
chcp 65001 > nul

echo =====================================================================
echo    🚀 HỆ THỐNG DỰ BÁO GIÁ MỞ CỬA - PIPELINE HUẤN LUYỆN THỦ CÔNG
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

echo [*] Đang bắt đầu chạy pipeline huấn luyện chéo và dự báo mới...
python scripts/run_training.py

echo.
echo =====================================================================
echo    🏆 PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG!
echo    - Xem log chi tiết tại: logs/predictions_history.txt
echo    - Xem biểu đồ so sánh tại: reports/figures/
echo =====================================================================
echo.
pause
