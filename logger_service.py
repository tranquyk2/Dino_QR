# ============================================================
#  logger_service.py — Lưu ảnh NG và ghi log OK vào CSV
# ============================================================
#
#  Sử dụng:
#    from logger_service import ResultLogger
#    logger = ResultLogger()
#    logger.log_ng(frame_bgr, result, lane="L1")
#    logger.log_ok(result, lane="L1")
#    logger.set_capture_dir("/path/to/captures")
#    logger.set_csv_path("/path/to/ok_log.csv")
# ============================================================

import os
import csv
import threading
import logging
from datetime import datetime

import cv2
import numpy as np

log = logging.getLogger(__name__)


class ResultLogger:
    """
    Thread-safe logger cho kết quả QR.
    - NG : chụp ảnh frame_bgr lưu vào capture_dir
    - OK : ghi 1 dòng vào file CSV (timestamp, lane, data, method)
    Thư mục và file CSV có thể thay đổi runtime qua set_capture_dir()
    và set_csv_path() (từ dialog Cài đặt).
    """

    # Header cố định cho CSV
    CSV_HEADER = ["timestamp", "lane", "qr_data", "method"]

    def __init__(self, capture_dir: str, csv_path: str,
                 enable_ng: bool = True, enable_ok: bool = True):
        self._lock        = threading.Lock()
        self._capture_dir = capture_dir
        self._csv_path    = csv_path
        self._enable_ng   = enable_ng
        self._enable_ok   = enable_ok

        # Tạo thư mục + file CSV nếu chưa có
        self._ensure_dirs()
        self._ensure_csv()

    # ----------------------------------------------------------
    # Cấu hình runtime (gọi từ dialog Cài đặt)
    # ----------------------------------------------------------
    def set_capture_dir(self, path: str):
        with self._lock:
            self._capture_dir = path
            os.makedirs(path, exist_ok=True)
        log.info(f"NG capture dir → {path}")

    def set_csv_path(self, path: str):
        with self._lock:
            self._csv_path = path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._ensure_csv()
        log.info(f"OK CSV log → {path}")

    def set_enable_ng(self, enabled: bool):
        self._enable_ng = enabled

    def set_enable_ok(self, enabled: bool):
        self._enable_ok = enabled

    @property
    def capture_dir(self) -> str:
        return self._capture_dir

    @property
    def csv_path(self) -> str:
        return self._csv_path

    # ----------------------------------------------------------
    # Ghi log NG — chụp ảnh
    # ----------------------------------------------------------
    def log_ng(self, frame_bgr: np.ndarray, lane: str = "L1") -> str | None:
        """
        Lưu ảnh NG vào capture_dir.
        Trả về đường dẫn file đã lưu, hoặc None nếu tắt / lỗi.
        """
        if not self._enable_ng:
            return None
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
        filename = os.path.join(self._capture_dir, f"NG_{lane}_{ts}.png")
        try:
            with self._lock:
                os.makedirs(self._capture_dir, exist_ok=True)
                cv2.imwrite(filename, frame_bgr)
            log.info(f"NG capture saved → {filename}")
            return filename
        except Exception as e:
            log.error(f"Không lưu được ảnh NG: {e}")
            return None

    # ----------------------------------------------------------
    # Ghi log OK — CSV
    # ----------------------------------------------------------
    def log_ok(self, qr_data: str, method: str = "", lane: str = "L1") -> bool:
        """
        Ghi 1 dòng OK vào CSV.
        Trả về True nếu thành công.
        """
        if not self._enable_ok:
            return False
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = [ts, lane, qr_data, method]
        try:
            with self._lock:
                os.makedirs(
                    os.path.dirname(self._csv_path) or ".", exist_ok=True
                )
                with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            return True
        except Exception as e:
            log.error(f"Không ghi được CSV OK: {e}")
            return False

    # ----------------------------------------------------------
    # Thống kê nhanh (hiển thị trên UI)
    # ----------------------------------------------------------
    def count_ng_files(self) -> int:
        """Đếm số file ảnh NG trong capture_dir."""
        try:
            return sum(
                1 for f in os.listdir(self._capture_dir)
                if f.startswith("NG_") and f.endswith(".png")
            )
        except Exception:
            return 0

    def count_ok_rows(self) -> int:
        """Đếm số dòng OK trong CSV (bỏ header)."""
        try:
            with open(self._csv_path, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return 0

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------
    def _ensure_dirs(self):
        os.makedirs(self._capture_dir, exist_ok=True)
        csv_dir = os.path.dirname(self._csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

    def _ensure_csv(self):
        """Tạo file CSV với header nếu chưa tồn tại."""
        try:
            path = self._csv_path
            csv_dir = os.path.dirname(path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self.CSV_HEADER)
        except Exception as e:
            log.error(f"Không tạo được CSV: {e}")