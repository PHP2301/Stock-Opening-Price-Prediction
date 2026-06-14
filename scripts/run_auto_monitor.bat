@echo off
:: Cấu hình UTF-8 để hiển thị tiếng Việt không bị lỗi font trên cmd Windows
chcp 65001 > nul
title 🚀 BACKGROUND STOCK MONITOR & AUTO PREDICTOR

echo =====================================================================
echo    🚀 HỆ THỐNG GIÁM SÁT TỰ ĐỘNG KHẨN CẤP & TỰ ĐỘNG DỰ BÁO CUỐI NGÀY
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

echo [*] Đang bắt đầu chạy Background Monitor tự động...
echo [!] Bạn hãy để cửa sổ này chạy ngầm để tự động quét tin tức sập mạng / bán tháo
echo     và gửi báo cáo Telegram hàng ngày.
echo.

python scripts/auto_monitor.py

echo.
echo =====================================================================
echo    ⚠️ TIẾN TRÌNH MONITOR ĐÃ DỪNG!
echo =====================================================================
pause
