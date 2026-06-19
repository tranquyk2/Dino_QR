# ============================================================
#  qr_detector.py — Logic phát hiện & decode QR code
#  [Tối ưu tốc độ phát hiện NG]
#  - Giảm NG_FRAME_THRESHOLD về config (khuyến nghị = 3)
#  - Early-exit: khi pyzbar IDLE, thử cv2.detect() nhanh trước
#    khi tăng ng_counter → phát hiện QR lỗi nhanh hơn ~1 frame
#  - Tách riêng _cv2_detect (QRCodeDetector đơn giản) để detect()
#    nhanh hơn QRCodeDetectorAruco (dùng ArUco finder phức tạp hơn)
#  - annotated frame copy chỉ khi cần vẽ, không copy thừa
# ============================================================

import cv2
import numpy as np
import time
import logging
import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

from config import (
    NG_FRAME_THRESHOLD, RESULT_HOLD_SECONDS,
    ENABLE_ROI, ROI_X, ROI_Y, ROI_W, ROI_H,
    ENABLE_ROI2, ROI2_X, ROI2_Y, ROI2_W, ROI2_H,
    ENABLE_PRESENCE_DETECTION, PRESENCE_STD_THRESHOLD,
    PRESENCE_NG_FRAME_THRESHOLD, DEBUG_PRINT_ROI_STD
)

logger = logging.getLogger(__name__)


# ── Enum kết quả ──────────────────────────────────────────────
class QRStatus(Enum):
    IDLE    = "IDLE"    # Chưa có QR trong vùng
    OK      = "OK"      # Decode thành công
    NG      = "NG"      # Detect được QR nhưng không decode được


@dataclass
class QRResult:
    status:    QRStatus         = QRStatus.IDLE
    data:      str              = ""
    bbox:      Optional[list]   = None   # [(x,y), ...] polygon
    timestamp: float            = field(default_factory=time.time)
    method:    str              = ""     # "pyzbar" hoặc "opencv"


# ── Detector chính ────────────────────────────────────────────
class QRDetector:
    """
    Phát hiện QR code trong mỗi frame.
    Chiến lược (theo thứ tự ưu tiên tốc độ):
      1. pyzbar decode → OK ngay nếu thành công
      2. Nếu pyzbar IDLE → cv2.detect() nhanh (không decode) để
         xác nhận có QR pattern → tăng ng_counter ngay lập tức
         thay vì chờ thêm 1 vòng qua detectAndDecodeMulti
      3. detectAndDecodeMulti / detectAndDecode làm fallback decode
      4. Nếu ng_counter >= NG_FRAME_THRESHOLD → báo NG
    """

    def __init__(self):
        # QRCodeDetectorAruco dùng ArUco finder pattern — decode tốt hơn
        # nhưng detect() chậm hơn QRCodeDetector đơn thuần.
        # Dùng 2 detector riêng:
        #   _cv2_qr      : ArUco-based, cho detectAndDecodeMulti
        #   _cv2_detect  : QRCodeDetector đơn, cho detect() nhanh
        if hasattr(cv2, "QRCodeDetectorAruco"):
            self._cv2_qr = cv2.QRCodeDetectorAruco()
        else:
            self._cv2_qr = cv2.QRCodeDetector()

        # Luôn dùng QRCodeDetector đơn cho detect() — nhẹ hơn ArUco
        self._cv2_detect = cv2.QRCodeDetector()

        self._ng_counter          = 0
        self._presence_ng_counter = 0
        self._last_result         = QRResult()
        self._hold_until          = 0.0
        self._last_debug_print    = 0.0

    # ----------------------------------------------------------
    def _has_object(self, roi_gray: np.ndarray) -> bool:
        """
        Kiểm tra trong vùng ROI có vật thể (texture) hay là nền trống.
        Dùng độ lệch chuẩn (std) của ảnh xám.
        """
        if roi_gray is None or roi_gray.size == 0:
            return False
        return float(np.std(roi_gray)) > PRESENCE_STD_THRESHOLD

    # ----------------------------------------------------------
    def _fast_detect(self, gray: np.ndarray) -> Optional[list]:
        """
        Chỉ detect (không decode) bằng QRCodeDetector đơn.
        Nhanh hơn detectAndDecodeMulti/ArUco ~2-3x.
        Trả về bbox list hoặc None.
        """
        try:
            found, bbox = self._cv2_detect.detect(gray)
            if found and bbox is not None:
                return bbox[0].astype(int).tolist()
        except Exception:
            pass
        return None

    # ----------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, frame_gray: np.ndarray,
                roi_offset: tuple = (0, 0)) -> tuple:
        """
        Xử lý 1 frame.
        frame_gray: ảnh xám đã crop + xử lý (CLAHE/blur) theo ROI.
        roi_offset: (rx, ry) vị trí gốc ROI trong frame_bgr.
        Trả về (QRResult, annotated_frame).
        """
        now = time.time()

        rx, ry = roi_offset
        rh, rw = frame_gray.shape[:2]

        if ENABLE_ROI:
            roi_bgr = frame_bgr[ry:ry + rh, rx:rx + rw]
        else:
            roi_bgr = frame_bgr

        roi_gray = frame_gray

        # Debug: in giá trị std của ROI mỗi ~1 giây
        if DEBUG_PRINT_ROI_STD and (now - self._last_debug_print) >= 1.0:
            std_val = float(np.std(roi_gray)) if roi_gray.size else 0.0
            print(f"[DEBUG] ROI std = {std_val:.2f}  "
                  f"(PRESENCE_STD_THRESHOLD = {PRESENCE_STD_THRESHOLD})")
            self._last_debug_print = now

        # -- Giai đoạn 1: Thử decode (chỉ trong ROI) -----------
        result = self._try_decode(roi_bgr, roi_gray)

        # Dịch bbox từ tọa độ ROI về tọa độ frame gốc
        if result.bbox:
            result.bbox = [(int(px) + rx, int(py) + ry)
                           for (px, py) in result.bbox]

        # -- Giai đoạn 2: Logic trạng thái ---------------------
        if result.status == QRStatus.OK:
            self._ng_counter          = 0
            self._presence_ng_counter = 0
            self._last_result         = result
            self._hold_until          = now + RESULT_HOLD_SECONDS

        elif result.status == QRStatus.NG:
            # Decode thất bại nhưng có QR pattern
            self._presence_ng_counter = 0
            self._ng_counter += 1
            if self._ng_counter >= NG_FRAME_THRESHOLD:
                self._last_result = result
                self._hold_until  = now + RESULT_HOLD_SECONDS

        else:
            # IDLE từ _try_decode — không decode được và pyzbar cũng IDLE.
            # Tuy nhiên pyzbar có thể bỏ sót QR bị lỗi nặng; thử fast detect
            # bằng cv2 để tăng ng_counter sớm hơn 1 vòng so với chờ
            # detectAndDecodeMulti ở vòng sau.
            fast_bbox = self._fast_detect(roi_gray)
            if fast_bbox is not None:
                # Phát hiện QR pattern nhưng không decode được
                translated_bbox = [(int(px) + rx, int(py) + ry)
                                   for (px, py) in fast_bbox]
                self._ng_counter += 1
                if self._ng_counter >= NG_FRAME_THRESHOLD:
                    self._last_result = QRResult(
                        status=QRStatus.NG,
                        bbox=translated_bbox,
                        method="fast-detect-only"
                    )
                    self._hold_until = now + RESULT_HOLD_SECONDS
                # Nếu chưa đủ threshold thì giữ _last_result cũ (hold)
            else:
                # Thực sự không có QR nào trong ROI
                self._ng_counter = 0

                # Kiểm tra presence (QR rách/mất finder pattern hoàn toàn)
                if ENABLE_PRESENCE_DETECTION and self._has_object(roi_gray):
                    self._presence_ng_counter += 1
                    if self._presence_ng_counter >= PRESENCE_NG_FRAME_THRESHOLD:
                        self._last_result = QRResult(
                            status=QRStatus.NG,
                            method="presence-only"
                        )
                        self._hold_until = now + RESULT_HOLD_SECONDS
                else:
                    self._presence_ng_counter = 0
                    if now > self._hold_until:
                        self._last_result = QRResult(status=QRStatus.IDLE)

        # -- Giai đoạn 3: Vẽ overlay (copy frame ở đây, không copy sớm) --
        annotated = frame_bgr.copy()
        annotated = self._draw_overlay(annotated, self._last_result)

        # -- Giai đoạn 4: Vẽ khung vùng quét (ROI) ------------
        if ENABLE_ROI:
            annotated = self._draw_roi(annotated, rx, ry, rw, rh)

        return self._last_result, annotated

    # ----------------------------------------------------------
    def _try_decode(self, frame_bgr: np.ndarray,
                    frame_gray: np.ndarray) -> QRResult:
        """
        Decode tất cả QR trong frame.
        Thứ tự: pyzbar (nhanh + chính xác với QR nhiễu) →
                detectAndDecodeMulti (multi QR) → detectAndDecode (fallback).
        Trả về OK / NG / IDLE.
        """

        # === Phương pháp 1: pyzbar ===
        if PYZBAR_AVAILABLE:
            decoded_objects = pyzbar.decode(
                frame_gray,
                symbols=[pyzbar.ZBarSymbol.QRCODE]
            )
            if decoded_objects:
                all_data = " | ".join(
                    obj.data.decode("utf-8", errors="replace")
                    for obj in decoded_objects
                )
                pts = [(p.x, p.y) for p in decoded_objects[0].polygon]
                return QRResult(
                    status=QRStatus.OK,
                    data=all_data,
                    bbox=pts,
                    method="pyzbar"
                )
            # pyzbar không decode được → trả IDLE, để process() gọi fast_detect
            # thay vì gọi _detect_qr_pattern() ở đây (tránh double detect)
            return QRResult(status=QRStatus.IDLE)

        # === Phương pháp 2: QRCodeDetectorAruco / detectAndDecodeMulti ===
        try:
            retval, decoded_list, points, _ = \
                self._cv2_qr.detectAndDecodeMulti(frame_gray)
            if retval and points is not None:
                ok_data  = [d for d in decoded_list if d]
                ng_count = sum(1 for d in decoded_list if not d)
                if ng_count > 0:
                    pts = points[0].astype(int).tolist()
                    return QRResult(
                        status=QRStatus.NG,
                        bbox=pts,
                        method="opencv-multi-detect-only"
                    )
                if ok_data:
                    all_data = " | ".join(ok_data)
                    pts = points[0].astype(int).tolist()
                    return QRResult(
                        status=QRStatus.OK,
                        data=all_data,
                        bbox=pts,
                        method="opencv-multi"
                    )
        except Exception as e:
            logger.debug(f"detectAndDecodeMulti error: {e}")

        # === Fallback: detectAndDecode đơn ===
        data, bbox, _ = self._cv2_qr.detectAndDecode(frame_gray)
        if data:
            pts = bbox[0].astype(int).tolist() if bbox is not None else None
            return QRResult(
                status=QRStatus.OK,
                data=data,
                bbox=pts,
                method="opencv"
            )
        elif bbox is not None:
            pts = bbox[0].astype(int).tolist()
            return QRResult(
                status=QRStatus.NG,
                bbox=pts,
                method="opencv-detect-only"
            )

        return QRResult(status=QRStatus.IDLE)

    # ----------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray,
                      result: QRResult,
                      banner_y_offset: int = 0) -> np.ndarray:
        """Vẽ bounding box và nhãn OK/NG lên frame."""
        h, w = frame.shape[:2]
        status = result.status

        COLOR_OK   = (0, 220, 100)
        COLOR_NG   = (0, 60,  220)
        COLOR_IDLE = (180, 180, 180)

        color = (COLOR_OK   if status == QRStatus.OK  else
                 COLOR_NG   if status == QRStatus.NG  else
                 COLOR_IDLE)

        if result.bbox:
            pts = np.array(result.bbox, dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 3)
            for pt in result.bbox:
                cv2.circle(frame, tuple(pt), 6, color, -1)

        if status != QRStatus.IDLE:
            label = "OK" if status == QRStatus.OK else "NG"
            banner_w, banner_h = 170, 60
            x0 = w - banner_w - 20
            y0 = 20 + banner_y_offset

            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0),
                          (x0 + banner_w, y0 + banner_h), color, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, label,
                        (x0 + 16, y0 + 44),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.5, (255, 255, 255), 2, cv2.LINE_AA)

            if status == QRStatus.NG and result.method in (
                    "presence-only", "fast-detect-only"):
                subtitle = ("QR bi loi"
                            if result.method == "presence-only"
                            else "QR bi loi")
                cv2.putText(frame, subtitle,
                            (x0 - 60, y0 + banner_h + 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, COLOR_NG, 1, cv2.LINE_AA)

        if status == QRStatus.OK and result.data:
            text = result.data[:60] + ("..." if len(result.data) > 60 else "")
            cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, f"Data: {text}",
                        (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 160), 1, cv2.LINE_AA)

        return frame

    # ----------------------------------------------------------
    def _draw_roi(self, frame: np.ndarray,
                  rx: int, ry: int, rw: int, rh: int,
                  label: str = "Vung quet QR",
                  color: tuple = (255, 165, 0)) -> np.ndarray:
        """Vẽ khung vùng quét (ROI) lên frame."""
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
        corner_len = 20
        thickness  = 3
        cv2.line(frame, (rx, ry),           (rx + corner_len, ry),           color, thickness)
        cv2.line(frame, (rx, ry),           (rx, ry + corner_len),           color, thickness)
        cv2.line(frame, (rx + rw, ry),      (rx + rw - corner_len, ry),      color, thickness)
        cv2.line(frame, (rx + rw, ry),      (rx + rw, ry + corner_len),      color, thickness)
        cv2.line(frame, (rx, ry + rh),      (rx + corner_len, ry + rh),      color, thickness)
        cv2.line(frame, (rx, ry + rh),      (rx, ry + rh - corner_len),      color, thickness)
        cv2.line(frame, (rx + rw, ry + rh), (rx + rw - corner_len, ry + rh), color, thickness)
        cv2.line(frame, (rx + rw, ry + rh), (rx + rw, ry + rh - corner_len), color, thickness)
        cv2.putText(frame, label, (rx, max(ry - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame

    # ----------------------------------------------------------
    def reset(self):
        self._ng_counter          = 0
        self._presence_ng_counter = 0
        self._last_result         = QRResult()
        self._hold_until          = 0.0

    @property
    def ng_frame_count(self) -> int:
        return self._ng_counter


# ── Detector đôi (2 vùng ROI song song) ─────────────────────────
class DualROIDetector:
    """
    Chạy 2 QRDetector độc lập song song bằng ThreadPoolExecutor.
    Mỗi detector giữ trạng thái riêng (ng_counter, hold_until, ...).
    """

    COLOR_ROI1 = (0, 165, 255)    # Cam (BGR)
    COLOR_ROI2 = (255, 200, 0)    # Xanh da trời đậm (BGR)

    def __init__(self):
        self._det1 = QRDetector()
        self._det2 = QRDetector()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="QRWorker"
        )

    # ----------------------------------------------------------
    def process_dual(
        self,
        frame_bgr: np.ndarray,
        gray1: np.ndarray,
        offset1: tuple,
        gray2: Optional[np.ndarray],
        offset2: Optional[tuple],
    ) -> tuple:
        """
        Xử lý 2 vùng ROI song song.
        Trả về (result1, result2, annotated_frame).
        result2 = None nếu ENABLE_ROI2 tắt.
        """
        fut1 = self._executor.submit(
            self._det1.process, frame_bgr, gray1, offset1
        )

        result2 = None
        fut2    = None
        if gray2 is not None and offset2 is not None:
            fut2 = self._executor.submit(
                self._det2.process, frame_bgr, gray2, offset2
            )

        result1, annotated = fut1.result()
        if fut2 is not None:
            result2, _ = fut2.result()
            annotated = self._det2._draw_overlay(
                annotated, result2, banner_y_offset=80
            )

        annotated = self._draw_dual_roi(annotated)
        return result1, result2, annotated

    # ----------------------------------------------------------
    def _draw_dual_roi(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ 2 khung ROI lên frame với màu riêng biệt."""
        h, w = frame.shape[:2]

        def _draw_box(frame, rx, ry, rw, rh, color, label):
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
            cl = 18
            for (cx, cy) in [
                (rx, ry), (rx + rw, ry), (rx, ry + rh), (rx + rw, ry + rh)
            ]:
                dx = 1 if cx == rx else -1
                dy = 1 if cy == ry else -1
                cv2.line(frame, (cx, cy), (cx + dx * cl, cy), color, 3)
                cv2.line(frame, (cx, cy), (cx, cy + dy * cl), color, 3)
            cv2.putText(frame, label, (rx, max(ry - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        if ENABLE_ROI:
            rx1 = max(0, min(int(w * ROI_X), w - 1))
            ry1 = max(0, min(int(h * ROI_Y), h - 1))
            rw1 = max(1, min(int(w * ROI_W), w - rx1))
            rh1 = max(1, min(int(h * ROI_H), h - ry1))
            _draw_box(frame, rx1, ry1, rw1, rh1, self.COLOR_ROI1, "Lane 1")

        if ENABLE_ROI2:
            rx2 = max(0, min(int(w * ROI2_X), w - 1))
            ry2 = max(0, min(int(h * ROI2_Y), h - 1))
            rw2 = max(1, min(int(w * ROI2_W), w - rx2))
            rh2 = max(1, min(int(h * ROI2_H), h - ry2))
            _draw_box(frame, rx2, ry2, rw2, rh2, self.COLOR_ROI2, "Lane 2")

        return frame

    # ----------------------------------------------------------
    def reset(self):
        self._det1.reset()
        self._det2.reset()

    def shutdown(self):
        """Dọn dẹp ThreadPoolExecutor khi đóng app."""
        self._executor.shutdown(wait=False)