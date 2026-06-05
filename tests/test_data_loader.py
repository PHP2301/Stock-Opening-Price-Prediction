import unittest
import pandas as pd
import numpy as np
import sys
import os

# Đường dẫn gốc dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_realtime_usd_vnd_rate

class TestDataLoader(unittest.TestCase):
    def test_usd_vnd_rate(self):
        """Kiểm tra tỷ giá USD/VND trả về có hợp lý không (nằm trong khoảng 20000 - 30000)"""
        rate = get_realtime_usd_vnd_rate()
        self.assertIsInstance(rate, float)
        self.assertTrue(20000.0 <= rate <= 30000.0, f"Tỷ giá {rate} nằm ngoài khoảng thực tế")

if __name__ == '__main__':
    unittest.main()
