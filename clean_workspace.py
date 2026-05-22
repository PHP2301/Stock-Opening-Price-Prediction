import os
import shutil
import sys

def clean_workspace():
    # Cấu hình encoding utf-8 cho Windows console
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # python < 3.7
        
    # Thư mục gốc của dự án (nơi đặt script này)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"=== Bắt đầu dọn dẹp thư mục: {root_dir} ===\n")
    
    # Các thư mục rác cần quét và xóa
    dirs_to_remove = {'__pycache__', '.ipynb_checkpoints', '.pytest_cache', '.ruff_cache'}
    # Các phần mở rộng tệp cần xóa
    extensions_to_remove = {'.pyc', '.pyo', '.pyd'}
    
    deleted_dirs_count = 0
    deleted_files_count = 0
    total_freed_bytes = 0
    
    # Duyệt qua cây thư mục
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Bỏ qua môi trường ảo và thư mục git để tránh xóa nhầm thư viện hoặc code nguồn
        if '.venv' in dirpath or '.git' in dirpath:
            continue
            
        # Xóa các thư mục rác chỉ định
        for dirname in list(dirnames):
            if dirname in dirs_to_remove:
                full_path = os.path.join(dirpath, dirname)
                try:
                    # Tính dung lượng giải phóng được
                    dir_size = 0
                    for r, d, files in os.walk(full_path):
                        for f in files:
                            fp = os.path.join(r, f)
                            if os.path.exists(fp):
                                dir_size += os.path.getsize(fp)
                                
                    shutil.rmtree(full_path)
                    rel_path = os.path.relpath(full_path, root_dir)
                    print(f"  [Xóa thư mục] {rel_path} ({dir_size / 1024:.2f} KB)")
                    deleted_dirs_count += 1
                    total_freed_bytes += dir_size
                except Exception as e:
                    print(f"  [Lỗi] Không xóa được thư mục {full_path}: {e}")
                    
        # Xóa các file có đuôi chỉ định
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions_to_remove:
                full_path = os.path.join(dirpath, filename)
                try:
                    file_size = os.path.getsize(full_path)
                    os.remove(full_path)
                    rel_path = os.path.relpath(full_path, root_dir)
                    print(f"  [Xóa file] {rel_path} ({file_size / 1024:.2f} KB)")
                    deleted_files_count += 1
                    total_freed_bytes += file_size
                except Exception as e:
                    print(f"  [Lỗi] Không xóa được file {full_path}: {e}")
                    
    print("\n==========================================")
    print("DỌN DẸP HOÀN TẤT!")
    print(f" - Tổng số thư mục đã xóa: {deleted_dirs_count}")
    print(f" - Tổng số file đã xóa: {deleted_files_count}")
    print(f" - Tổng dung lượng giải phóng: {total_freed_bytes / (1024 * 1024):.2f} MB")
    print("==========================================")

if __name__ == '__main__':
    clean_workspace()
