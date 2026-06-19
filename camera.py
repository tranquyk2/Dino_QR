# ============================================================
#  camera.py — Quản lý camera Dino-Lite qua OpenCV
# ============================================================

import cv2
import threading
import time
import numpy as np
import logging
from config import (
    CAMERA_INDEX, CAMERA_BACKEND,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    ENABLE_CLAHE, CLAHE_CLIP, CLAHE_GRID,
    ENABLE_ROI, ROI_X, ROI_Y, ROI_W, ROI_H,
    ENABLE_ROI2, ROI2_X, ROI2_Y, ROI2_W, ROI2_H,
)

logger = logging.getLogger(__name__)


def list_available_cameras(max_test: int = 6) -> list[int]:
    """Liệt kê tất cả camera index đang hoạt động."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


class DinoCamera:
    """
    Thread-safe camera manager cho Dino-Lite (UVC).
    Đọc frame liên tục trong background thread.
    """

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP,
            tileGridSize=CLAHE_GRID
        ) if ENABLE_CLAHE else None

    # ----------------------------------------------------------
    def open(self, index: int = CAMERA_INDEX) -> bool:
        """Mở camera theo index. Trả về True nếu thành công."""
        backend = {
            "DSHOW": cv2.CAP_DSHOW,
            "MSMF":  cv2.CAP_MSMF,
            "AUTO":  cv2.CAP_ANY,
        }.get(CAMERA_BACKEND, cv2.CAP_DSHOW)

        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            logger.error(f"Không mở được camera index={index}")
            return False

        # Thiết lập độ phân giải
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        # Giảm buffer để giảm độ trễ
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Kiểm tra độ phân giải thực tế (camera có thể không hỗ trợ giá trị yêu cầu)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Yêu cầu {CAMERA_WIDTH}x{CAMERA_HEIGHT}, thực tế {actual_w}x{actual_h}")

        # Thử đọc 1 frame test để xác nhận camera hoạt động
        ret, test_frame = self._cap.read()
        if not ret or test_frame is None:
            logger.warning(
                f"Camera index={index} mở được nhưng đọc frame test thất bại "
                f"({actual_w}x{actual_h}). Thử lại với độ phân giải mặc định."
            )
            # Fallback: bỏ thiết lập độ phân giải, dùng mặc định của camera
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, test_frame = self._cap.read()
            if not ret or test_frame is None:
                logger.error(f"Camera index={index} không đọc được frame nào sau khi fallback.")
                self._cap.release()
                return False
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Fallback thành công, độ phân giải hiện tại {actual_w}x{actual_h}")

        # Lưu frame test đầu tiên để vòng lặp hiển thị có dữ liệu ngay
        with self._lock:
            self._frame = test_frame

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraThread"
        )
        self._thread.start()
        logger.info(f"Camera index={index} đã mở thành công.")
        return True

    # ----------------------------------------------------------
    def _capture_loop(self):
        """Vòng lặp đọc frame chạy trong background."""
        fail_count = 0
        while self._running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    fail_count = 0
                    with self._lock:
                        self._frame = frame
                else:
                    fail_count += 1
                    if fail_count % 30 == 0:
                        logger.warning(f"Đọc frame thất bại liên tiếp {fail_count} lần.")
                    time.sleep(0.01)
            else:
                break

    # ----------------------------------------------------------
    def get_frame(self) -> np.ndarray | None:
        """Lấy frame mới nhất (thread-safe)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    # ----------------------------------------------------------
    def get_processed_frame(self):
        """
        Trả về (frame_bgr_original, frame_gray_enhanced, roi_offset).

        frame_gray_enhanced: ảnh xám đã xử lý (CLAHE, blur), CHỈ chứa
        vùng ROI (nếu ENABLE_ROI) để giảm tải xử lý.
        roi_offset: (rx, ry) - vị trí gốc của vùng ROI trong frame gốc,
        dùng để dịch tọa độ bbox kết quả về frame gốc.
        Nếu không bật ROI, roi_offset = (0, 0) và gray = toàn frame.
        """
        frame = self.get_frame()
        if frame is None:
            return None, None, (0, 0)

        h, w = frame.shape[:2]

        if ENABLE_ROI:
            rx = max(0, min(int(w * ROI_X), w - 1))
            ry = max(0, min(int(h * ROI_Y), h - 1))
            rw = max(1, min(int(w * ROI_W), w - rx))
            rh = max(1, min(int(h * ROI_H), h - ry))
            roi_bgr = frame[ry:ry + rh, rx:rx + rw]
        else:
            rx, ry = 0, 0
            roi_bgr = frame

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        if self._clahe is not None:
            gray = self._clahe.apply(gray)

        # Khử nhiễu nhẹ
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return frame, gray, (rx, ry)

    # ----------------------------------------------------------
    def get_processed_frame_dual(self):
        """
        Trả về (frame_bgr, gray1, offset1, gray2, offset2) cho 2 vùng ROI song song.

        gray1, offset1: vùng ROI 1 (ENABLE_ROI / ROI_X,Y,W,H)
        gray2, offset2: vùng ROI 2 (ENABLE_ROI2 / ROI2_X,Y,W,H)
        Nếu ROI2 tắt, gray2=None, offset2=None.
        """
        frame = self.get_frame()
        if frame is None:
            return None, None, (0, 0), None, None

        h, w = frame.shape[:2]

        def _extract_roi(frame, x_ratio, y_ratio, w_ratio, h_ratio):
            rx = max(0, min(int(w * x_ratio), w - 1))
            ry = max(0, min(int(h * y_ratio), h - 1))
            rw = max(1, min(int(w * w_ratio), w - rx))
            rh = max(1, min(int(h * h_ratio), h - ry))
            roi_bgr = frame[ry:ry + rh, rx:rx + rw]
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            if self._clahe is not None:
                gray = self._clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            return gray, (rx, ry)

        if ENABLE_ROI:
            gray1, offset1 = _extract_roi(frame, ROI_X, ROI_Y, ROI_W, ROI_H)
        else:
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._clahe:
                gray1 = self._clahe.apply(gray1)
            gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
            offset1 = (0, 0)

        if ENABLE_ROI2:
            gray2, offset2 = _extract_roi(frame, ROI2_X, ROI2_Y, ROI2_W, ROI2_H)
        else:
            gray2, offset2 = None, None

        return frame, gray1, offset1, gray2, offset2

    # ----------------------------------------------------------
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ----------------------------------------------------------
    def release(self):
        """Dừng thread và giải phóng camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("Camera đã được giải phóng.")

    # ----------------------------------------------------------
    def get_resolution(self) -> tuple[int, int]:
        if not self._cap:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)