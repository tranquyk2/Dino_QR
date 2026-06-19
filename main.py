# ============================================================
#  main.py — Giao diện GUI và vòng lặp chính
# ============================================================

import os
import sys
import time
import threading
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from datetime import datetime
import cv2
import numpy as np
import customtkinter as ctk
import tkinter.messagebox as msgbox
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk

from camera       import DinoCamera, list_available_cameras
from qr_detector  import DualROIDetector, QRDetector, QRStatus, QRResult
from arduino_sent import ArduinoConnection, list_serial_ports, find_arduino_port
from roi_editor   import ROIEditorDialog
from logger_service import ResultLogger
from config import (
    APP_TITLE, WINDOW_W, WINDOW_H, THEME,
    LOG_DIR, CAPTURE_DIR, CSV_PATH,
    RESULT_HOLD_SECONDS,
    ROI_X, ROI_Y, ROI_W, ROI_H,
    ENABLE_NG_CAPTURE, ENABLE_OK_CSV_LOG,
    save_paths_to_config,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# GUI Application
# -----------------------------------------------------------------
class QRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        ctk.set_appearance_mode(THEME)
        ctk.set_default_color_theme("dark-blue")

        # ── ResultLogger ─────────────────────────────────────────
        self.result_logger = ResultLogger(
            capture_dir = CAPTURE_DIR,
            csv_path    = CSV_PATH,
            enable_ng   = ENABLE_NG_CAPTURE,
            enable_ok   = ENABLE_OK_CSV_LOG,
        )

        # Biến đếm hiển thị trên UI (cập nhật sau mỗi lần log)
        self._ng_count = self.result_logger.count_ng_files()
        self._ok_count = self.result_logger.count_ok_rows()

        # Chống ghi ảnh NG trùng lặp trong 1 lần hold
        self._last_ng_saved: dict[str, float] = {}   # lane → timestamp

        # ── Top bar ──────────────────────────────────────────────
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(top_frame, text="Camera:", width=60).pack(side="left", padx=5)
        self.cam_option = ctk.CTkOptionMenu(
            top_frame, values=[], command=self._on_camera_change)
        self.cam_option.pack(side="left", padx=5)

        # Bộ đếm NG / OK
        self._ng_var = ctk.StringVar(value=f"NG: {self._ng_count}")
        self._ok_var = ctk.StringVar(value=f"OK: {self._ok_count}")
        ctk.CTkLabel(top_frame, textvariable=self._ng_var,
                     text_color="#FF4444", width=70).pack(side="left", padx=4)
        ctk.CTkLabel(top_frame, textvariable=self._ok_var,
                     text_color="#44CC44", width=70).pack(side="left", padx=4)

        ctk.CTkButton(top_frame, text="Cài đặt",
                      command=self._open_settings).pack(side="right", padx=5)
        ctk.CTkButton(top_frame, text="✏ Chỉnh ROI",
                      fg_color="#1a5276", hover_color="#21618c",
                      command=self._open_roi_editor).pack(side="right", padx=5)
        ctk.CTkButton(top_frame, text="Test NG",
                      fg_color="#8B0000", hover_color="#a00000",
                      command=self._test_ng).pack(side="right", padx=5)

        self.arduino_status_var = ctk.StringVar(value="Arduino: …")
        ctk.CTkLabel(top_frame, textvariable=self.arduino_status_var,
                     width=160).pack(side="right", padx=5)

        # ── Camera init ──────────────────────────────────────────
        self._populate_camera_options()
        self.camera = DinoCamera()
        self.camera_index = int(self.cam_option.get()) if self.cam_option.get() else 0
        opened = self.camera.open(self.camera_index)

        # ── Video area ───────────────────────────────────────────
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        video_frame = ctk.CTkFrame(main_frame)
        video_frame.pack(side="left", fill="both", expand=True)
        self.video_label = ctk.CTkLabel(video_frame, text="Chuẩn bị video…")
        self.video_label.pack(fill="both", expand=True)

        self.status_var = ctk.StringVar(value="Khởi động…")
        self.status_bar = ctk.CTkLabel(self, textvariable=self.status_var, height=24)
        self.status_bar.pack(fill="x", side="bottom")

        # Fallback camera
        if not opened:
            for idx in range(1, 5):
                self.camera = DinoCamera()
                if self.camera.open(idx):
                    opened = True
                    self.camera_index = idx
                    break

        if not opened:
            self.status_var.set("Không mở được camera. Kiểm tra kết nối.")
            logger.error(f"Camera open failed for index {self.camera_index}")
            return

        self.status_var.set(f"Camera mở thành công (index {self.camera_index}).")

        # ── Detector & Arduino ───────────────────────────────────
        self.detector = DualROIDetector()
        self.arduino  = ArduinoConnection(auto_detect=True, auto_reconnect=True)
        if self.arduino.is_connected():
            logger.info(f"Arduino kết nối tại {self.arduino.port}")
        else:
            logger.info("Chưa tìm thấy Arduino. Sẽ tự kết nối khi cắm vào.")

        # ── Background thread ────────────────────────────────────
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._update_arduino_status()

    # -----------------------------------------------------------------
    def _update_arduino_status(self):
        if self.arduino.is_connected():
            self.arduino_status_var.set(f"🟢 Arduino: {self.arduino.port}")
        else:
            self.arduino_status_var.set("🔴 Arduino: chưa kết nối")
        self.after(1000, self._update_arduino_status)

    # -----------------------------------------------------------------
    def _run_loop(self):
        no_frame_count    = 0
        last_display_time = 0.0
        DISPLAY_INTERVAL  = 1.0 / 20.0   # Hiển thị tối đa 20 FPS
        PROCESS_INTERVAL  = 1.0 / 30.0   # Xử lý tối đa 30 FPS
        last_process_time = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            # Throttle vòng xử lý — tránh CPU 100%
            elapsed = now - last_process_time
            if elapsed < PROCESS_INTERVAL:
                time.sleep(PROCESS_INTERVAL - elapsed)
                continue
            last_process_time = time.time()

            frame_bgr, gray1, offset1, gray2, offset2 = \
                self.camera.get_processed_frame_dual()

            if frame_bgr is None:
                no_frame_count += 1
                if no_frame_count == 60:
                    self.after(0, lambda: self.status_var.set(
                        "⚠️ Không nhận được hình ảnh từ camera. "
                        "Kiểm tra camera hoặc đổi độ phân giải trong config.py."
                    ))
                time.sleep(0.05)
                continue
            no_frame_count = 0

            result1, result2, annotated = self.detector.process_dual(
                frame_bgr, gray1, offset1, gray2, offset2
            )

            # Lưu ảnh NG / CSV OK (chạy trong thread này, không block UI)
            self._do_logging(frame_bgr, result1, "L1")
            if result2 is not None:
                self._do_logging(frame_bgr, result2, "L2")

            # Throttle hiển thị ~20 FPS
            now = time.time()
            if now - last_display_time >= DISPLAY_INTERVAL:
                last_display_time = now
                display_frame = cv2.resize(
                    annotated, (WINDOW_W, WINDOW_H),
                    interpolation=cv2.INTER_AREA
                )
                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                self.after(0, self._display_frame, rgb)

            if result1.status != QRStatus.IDLE or \
               (result2 is not None and result2.status != QRStatus.IDLE):
                self.after(0, self._handle_detection, result1, result2, frame_bgr)

    # -----------------------------------------------------------------
    def _do_logging(self, frame_bgr: np.ndarray,
                    result: QRResult, lane: str):
        """
        Gọi ResultLogger cho NG (ảnh) và OK (CSV).
        Dùng _last_ng_saved để chống lưu trùng trong cùng 1 lần hold.
        """
        now = time.time()

        if result.status == QRStatus.NG:
            last_saved = self._last_ng_saved.get(lane, 0.0)
            # Chỉ chụp 1 lần mỗi 2 giây (= RESULT_HOLD_SECONDS) per lane
            if now - last_saved >= RESULT_HOLD_SECONDS:
                path = self.result_logger.log_ng(frame_bgr, lane=lane)
                if path:
                    self._last_ng_saved[lane] = now
                    self._ng_count += 1
                    self.after(0, self._update_counters)

        elif result.status == QRStatus.OK:
            # Ghi CSV mỗi lần decode thành công (đã được debounce bởi RESULT_HOLD)
            last_ok = self._last_ng_saved.get(f"ok_{lane}", 0.0)
            if now - last_ok >= RESULT_HOLD_SECONDS:
                self.result_logger.log_ok(
                    qr_data=result.data,
                    method=result.method,
                    lane=lane
                )
                self._last_ng_saved[f"ok_{lane}"] = now
                self._ok_count += 1
                self.after(0, self._update_counters)

    # -----------------------------------------------------------------
    def _update_counters(self):
        self._ng_var.set(f"NG: {self._ng_count}")
        self._ok_var.set(f"OK: {self._ok_count}")

    # -----------------------------------------------------------------
    def _display_frame(self, rgb_frame: np.ndarray):
        pil    = Image.fromarray(rgb_frame)
        old    = getattr(self.video_label, "image", None)
        tk_img = ImageTk.PhotoImage(pil)
        self.video_label.configure(image=tk_img, text="")
        self.video_label.image = tk_img
        if old is not None:
            del old

    # -----------------------------------------------------------------
    def _handle_detection(self, result1: QRResult,
                          result2: QRResult = None,
                          frame_bgr: np.ndarray = None):
        def _fmt(r: QRResult, lane: str) -> str:
            if r.status == QRStatus.OK:
                text = r.data[:30] + "..." if len(r.data) > 30 else r.data
                return f"✅ {lane}: OK [{text}]"
            elif r.status == QRStatus.NG:
                return f"❌ {lane}: NG"
            return f"⏳ {lane}: Scanning…"

        parts = [_fmt(result1, "L1")]
        if result2 is not None:
            parts.append(_fmt(result2, "L2"))
        self.status_var.set("  |  ".join(parts))

        statuses = [result1.status]
        if result2 is not None:
            statuses.append(result2.status)

        if QRStatus.NG in statuses:
            self.status_bar.configure(fg_color="#8B0000")
            self.arduino.send_ng()
        elif QRStatus.OK in statuses:
            self.status_bar.configure(fg_color="#006400")
            self.arduino.send_ok()
        else:
            self.status_bar.configure(fg_color="transparent")

    # -----------------------------------------------------------------
    def _test_ng(self):
        fake = QRResult(status=QRStatus.NG, method="manual-test")
        self._handle_detection(fake, None)

    # -----------------------------------------------------------------
    def _open_roi_editor(self):
        if not hasattr(self, "camera") or not self.camera.is_opened():
            self.status_var.set("Camera chưa sẵn sàng — không thể chỉnh ROI.")
            return
        import importlib, config as _cfg
        importlib.reload(_cfg)
        initial = {
            "ROI_X":  _cfg.ROI_X,  "ROI_Y":  _cfg.ROI_Y,
            "ROI_W":  _cfg.ROI_W,  "ROI_H":  _cfg.ROI_H,
            "ROI2_X": _cfg.ROI2_X, "ROI2_Y": _cfg.ROI2_Y,
            "ROI2_W": _cfg.ROI2_W, "ROI2_H": _cfg.ROI2_H,
        }
        ROIEditorDialog(self, self.camera, initial)

    # -----------------------------------------------------------------
    def on_close(self):
        self._stop_event.set()
        self.camera.release()
        if hasattr(self, "detector"):
            self.detector.shutdown()
        if hasattr(self, "arduino"):
            self.arduino.close()
        self.destroy()

    # -----------------------------------------------------------------
    def _populate_camera_options(self):
        try:
            cams = list_available_cameras()
        except Exception as e:
            logger.warning(f"Lỗi khi liệt kê camera: {e}")
            cams = []
        cam_strs = [str(i) for i in cams] or ["0"]
        self.cam_option.configure(values=cam_strs)
        self.cam_option.set(cam_strs[0])

    def _on_camera_change(self, selected_index: str):
        idx = int(selected_index)
        self.cam_option.configure(state="disabled")
        self.status_var.set(f"Đang chuyển sang camera {idx}…")
        threading.Thread(target=self._switch_camera,
                         args=(idx,), daemon=True).start()

    def _switch_camera(self, idx: int):
        old_camera = self.camera
        new_camera = DinoCamera()
        opened     = new_camera.open(idx)
        if opened:
            self.camera       = new_camera
            self.camera_index = idx
            old_camera.release()
            self.after(0, lambda: self.status_var.set(
                f"Camera mở thành công (index {idx})."))
        else:
            new_camera.release()
            self.after(0, lambda: self.status_var.set(
                f"Không mở được camera index {idx}."))
        self.after(0, lambda: self.cam_option.configure(state="normal"))

    # =================================================================
    # Dialog Cài đặt (Arduino + Logging paths)
    # =================================================================
    def _open_settings(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Cài đặt")
        dlg.geometry("540x480")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        # Notebook-style tabs
        tab = ctk.CTkTabview(dlg, width=520, height=440)
        tab.pack(fill="both", expand=True, padx=10, pady=10)
        tab.add("Arduino")
        tab.add("Lưu trữ")

        # ── Tab Arduino ───────────────────────────────────────────
        self._build_arduino_tab(tab.tab("Arduino"))

        # ── Tab Lưu trữ ──────────────────────────────────────────
        self._build_storage_tab(tab.tab("Lưu trữ"), dlg)

    # -----------------------------------------------------------------
    def _build_arduino_tab(self, parent):
        pad = {"padx": 14, "pady": 7}

        ctk.CTkLabel(parent, text="Cổng COM:").grid(
            row=0, column=0, sticky="w", **pad)
        available_ports = list_serial_ports() or [self.arduino.port]
        port_var  = ctk.StringVar(value=self.arduino.port)
        port_menu = ctk.CTkOptionMenu(
            parent, values=available_ports, variable=port_var, width=160)
        port_menu.grid(row=0, column=1, sticky="w", **pad)

        def _refresh():
            ports = list_serial_ports() or [port_var.get()]
            port_menu.configure(values=ports)

        ctk.CTkButton(parent, text="Quét lại", width=70,
                      command=_refresh).grid(row=0, column=2, **pad)

        ctk.CTkLabel(parent, text="Baudrate:").grid(
            row=1, column=0, sticky="w", **pad)
        baud_var  = ctk.StringVar(value=str(self.arduino.baudrate))
        baud_menu = ctk.CTkOptionMenu(
            parent,
            values=["9600", "19200", "38400", "57600", "115200"],
            variable=baud_var, width=160)
        baud_menu.grid(row=1, column=1, sticky="w", **pad)

        ard_status = ctk.StringVar(value=(
            f"Đang kết nối: {self.arduino.port}"
            if self.arduino.is_connected() else "Chưa kết nối Arduino"))
        ctk.CTkLabel(parent, textvariable=ard_status,
                     wraplength=380, justify="left").grid(
            row=2, column=0, columnspan=3, sticky="w", **pad)

        def _connect():
            port = port_var.get().strip()
            try:
                baud = int(baud_var.get())
            except ValueError:
                ard_status.set("Lỗi: Baudrate phải là số nguyên.")
                return
            ok = self.arduino.reconnect(port=port, baudrate=baud)
            ard_status.set(
                f"Kết nối thành công: {port} @ {baud} baud" if ok
                else f"Không kết nối được tới {port}.")

        def _test():
            resp = self.arduino.request_status()
            ard_status.set(
                f"Phản hồi: {resp}" if resp
                else "Không nhận được phản hồi từ Arduino.")

        def _auto():
            found = find_arduino_port()
            if found:
                port_var.set(found)
                _refresh()
                ard_status.set(
                    f"Đã tìm thấy Arduino tại {found}. Nhấn 'Kết nối'.")
            else:
                ard_status.set(
                    "Không tìm thấy Arduino (kiểm tra VID/PID, cáp, driver).")

        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.grid(row=3, column=0, columnspan=3, pady=(10, 6))
        ctk.CTkButton(bf, text="Tự động dò",    command=_auto,    width=100).pack(side="left", padx=6)
        ctk.CTkButton(bf, text="Kết nối",        command=_connect, width=100).pack(side="left", padx=6)
        ctk.CTkButton(bf, text="Test trạng thái", command=_test,   width=110).pack(side="left", padx=6)

    # -----------------------------------------------------------------
    def _build_storage_tab(self, parent, dlg):
        """Tab cài đặt thư mục lưu ảnh NG và file CSV OK."""
        pad = {"padx": 14, "pady": 8}

        # ── Bật / tắt ─────────────────────────────────────────────
        ng_enable_var = ctk.BooleanVar(value=self.result_logger._enable_ng)
        ok_enable_var = ctk.BooleanVar(value=self.result_logger._enable_ok)

        ctk.CTkCheckBox(parent, text="Lưu ảnh khi phát hiện NG",
                        variable=ng_enable_var).grid(
            row=0, column=0, columnspan=3, sticky="w", **pad)
        ctk.CTkCheckBox(parent, text="Ghi log OK vào CSV",
                        variable=ok_enable_var).grid(
            row=1, column=0, columnspan=3, sticky="w", **pad)

        # Divider
        ctk.CTkLabel(parent, text="─" * 52,
                     text_color="gray").grid(
            row=2, column=0, columnspan=3, pady=(4, 0))

        # ── Thư mục lưu ảnh NG ───────────────────────────────────
        ctk.CTkLabel(parent, text="📁 Thư mục ảnh NG:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, sticky="w", **pad)

        capture_var = ctk.StringVar(value=self.result_logger.capture_dir)
        capture_entry = ctk.CTkEntry(parent, textvariable=capture_var,
                                     width=310, state="readonly")
        capture_entry.grid(row=4, column=0, columnspan=2, sticky="w",
                           padx=14, pady=2)

        def _browse_capture():
            chosen = filedialog.askdirectory(
                title="Chọn thư mục lưu ảnh NG",
                initialdir=capture_var.get() or os.path.expanduser("~"),
                parent=dlg,
            )
            if chosen:
                capture_var.set(chosen)

        ctk.CTkButton(parent, text="Chọn…", width=80,
                      command=_browse_capture).grid(
            row=4, column=2, padx=8, pady=2)

        # Số ảnh NG hiện tại
        ng_count_var = ctk.StringVar(
            value=f"Hiện có {self.result_logger.count_ng_files()} ảnh NG")
        ctk.CTkLabel(parent, textvariable=ng_count_var,
                     text_color="gray").grid(
            row=5, column=0, columnspan=3, sticky="w", padx=14)

        # Nút mở thư mục
        def _open_capture_folder():
            path = capture_var.get()
            if os.path.isdir(path):
                os.startfile(path)   # Windows
            else:
                msgbox.showwarning("Thư mục không tồn tại",
                                   f"Chưa tạo thư mục:\n{path}", parent=dlg)

        ctk.CTkButton(parent, text="📂 Mở thư mục", width=120,
                      fg_color="#1a5276", hover_color="#21618c",
                      command=_open_capture_folder).grid(
            row=6, column=0, sticky="w", padx=14, pady=(2, 8))

        # ── File CSV OK ───────────────────────────────────────────
        ctk.CTkLabel(parent, text="📄 File CSV log OK:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=7, column=0, sticky="w", **pad)

        csv_var = ctk.StringVar(value=self.result_logger.csv_path)
        csv_entry = ctk.CTkEntry(parent, textvariable=csv_var,
                                 width=310, state="readonly")
        csv_entry.grid(row=8, column=0, columnspan=2, sticky="w",
                       padx=14, pady=2)

        def _browse_csv():
            chosen = filedialog.asksaveasfilename(
                title="Chọn / tạo file CSV log OK",
                initialfile=os.path.basename(csv_var.get()),
                initialdir=os.path.dirname(csv_var.get())
                           or os.path.expanduser("~"),
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                parent=dlg,
            )
            if chosen:
                csv_var.set(chosen)

        ctk.CTkButton(parent, text="Chọn…", width=80,
                      command=_browse_csv).grid(
            row=8, column=2, padx=8, pady=2)

        # Số dòng OK hiện tại
        ok_count_var = ctk.StringVar(
            value=f"Hiện có {self.result_logger.count_ok_rows()} bản ghi OK")
        ctk.CTkLabel(parent, textvariable=ok_count_var,
                     text_color="gray").grid(
            row=9, column=0, columnspan=3, sticky="w", padx=14)

        def _open_csv():
            path = csv_var.get()
            if os.path.isfile(path):
                os.startfile(path)
            else:
                msgbox.showwarning("File chưa tồn tại",
                                   f"Chưa có file:\n{path}", parent=dlg)

        ctk.CTkButton(parent, text="📄 Mở CSV", width=100,
                      fg_color="#145a32", hover_color="#1e8449",
                      command=_open_csv).grid(
            row=10, column=0, sticky="w", padx=14, pady=(2, 8))

        # ── Nút Lưu / Đóng ───────────────────────────────────────
        save_status = ctk.StringVar(value="")
        ctk.CTkLabel(parent, textvariable=save_status,
                     text_color="#44CC44").grid(
            row=11, column=0, columnspan=3, pady=(4, 0))

        def _save():
            new_capture = capture_var.get().strip()
            new_csv     = csv_var.get().strip()

            if not new_capture:
                msgbox.showwarning("Thiếu thông tin",
                                   "Vui lòng chọn thư mục lưu ảnh NG.",
                                   parent=dlg)
                return
            if not new_csv:
                msgbox.showwarning("Thiếu thông tin",
                                   "Vui lòng chọn file CSV.", parent=dlg)
                return

            # Áp dụng runtime
            self.result_logger.set_capture_dir(new_capture)
            self.result_logger.set_csv_path(new_csv)
            self.result_logger.set_enable_ng(ng_enable_var.get())
            self.result_logger.set_enable_ok(ok_enable_var.get())

            # Ghi vào config.py để giữ sau khi restart
            saved = save_paths_to_config(new_capture, new_csv)

            # Cập nhật bộ đếm
            ng_count_var.set(
                f"Hiện có {self.result_logger.count_ng_files()} ảnh NG")
            ok_count_var.set(
                f"Hiện có {self.result_logger.count_ok_rows()} bản ghi OK")

            save_status.set(
                "✅ Đã lưu cài đặt!" if saved
                else "⚠️ Áp dụng runtime OK, không ghi được config.py.")

        bf2 = ctk.CTkFrame(parent, fg_color="transparent")
        bf2.grid(row=12, column=0, columnspan=3, pady=(8, 6))
        ctk.CTkButton(bf2, text="💾 Lưu cài đặt", command=_save,
                      width=130, fg_color="#1a5276",
                      hover_color="#21618c").pack(side="left", padx=8)
        ctk.CTkButton(bf2, text="Đóng", command=dlg.destroy,
                      width=80).pack(side="left", padx=8)


# -----------------------------------------------------------------
if __name__ == "__main__":
    app = QRApp()
    app.mainloop()