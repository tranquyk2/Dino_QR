# ============================================================
#  roi_editor.py — Dialog chỉnh vùng ROI trực tiếp trên ảnh camera
# ============================================================

import cv2
import numpy as np
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk


class ROIEditorDialog(ctk.CTkToplevel):
    """
    Hiển thị frame camera và cho phép kéo-thả để điều chỉnh
    vị trí / kích thước của 1 hoặc 2 vùng ROI.
    Khi nhấn "Lưu", ghi giá trị mới vào config.py.
    """

    COLOR_ROI1 = (0, 165, 255)   # Cam (BGR)
    COLOR_ROI2 = (255, 200, 0)   # Xanh da trời (BGR)

    def __init__(self, parent, camera, initial: dict):
        super().__init__(parent)
        self.title("Chỉnh vùng quét ROI")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._camera   = camera
        self._running  = True

        # Kích thước canvas hiển thị
        self._cw = 960
        self._ch = 540

        # Trạng thái ROI (tỉ lệ 0-1)
        self._roi = {
            "ROI_X":  initial.get("ROI_X",  0.2425),
            "ROI_Y":  initial.get("ROI_Y",  0.1050),
            "ROI_W":  initial.get("ROI_W",  0.4951),
            "ROI_H":  initial.get("ROI_H",  0.5183),
            "ROI2_X": initial.get("ROI2_X", 0.55),
            "ROI2_Y": initial.get("ROI2_Y", 0.1050),
            "ROI2_W": initial.get("ROI2_W", 0.4951),
            "ROI2_H": initial.get("ROI2_H", 0.5183),
        }

        # Import config để biết ROI2 có bật không
        import config as _cfg
        self._enable_roi2 = _cfg.ENABLE_ROI2

        # Drag state
        self._drag_target = None   # None | "roi1" | "roi2"
        self._drag_mode   = None   # "move" | "resize"
        self._drag_start  = (0, 0)
        self._drag_orig   = {}

        self._build_ui()
        self._update_frame()

    # ----------------------------------------------------------
    def _build_ui(self):
        # Canvas hiển thị camera
        self._canvas = tk.Canvas(self, width=self._cw, height=self._ch,
                                 bg="black", cursor="crosshair")
        self._canvas.pack(padx=8, pady=8)
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",        self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

        # Spinbox controls
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(fill="x", padx=8, pady=(0, 4))

        # ROI 1
        ctk.CTkLabel(ctrl, text="── Lane 1 ──", text_color="#FFA500").grid(
            row=0, column=0, columnspan=8, sticky="w", padx=4, pady=2)

        self._vars1 = {}
        for col, key in enumerate(["ROI_X", "ROI_Y", "ROI_W", "ROI_H"]):
            label = key.replace("ROI_", "")
            ctk.CTkLabel(ctrl, text=f"{label}:", width=28).grid(row=1, column=col*2, padx=2)
            var = ctk.StringVar(value=f"{self._roi[key]:.4f}")
            self._vars1[key] = var
            entry = ctk.CTkEntry(ctrl, textvariable=var, width=72)
            entry.grid(row=1, column=col*2+1, padx=2)
            entry.bind("<FocusOut>", lambda e, k=key: self._on_entry_change(k, 1))
            entry.bind("<Return>",   lambda e, k=key: self._on_entry_change(k, 1))

        # ROI 2
        self._roi2_enabled_var = ctk.BooleanVar(value=self._enable_roi2)
        ctk.CTkCheckBox(ctrl, text="Bật Lane 2",
                        variable=self._roi2_enabled_var,
                        command=self._on_toggle_roi2).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 2))

        self._vars2 = {}
        self._roi2_entries = []
        for col, key in enumerate(["ROI2_X", "ROI2_Y", "ROI2_W", "ROI2_H"]):
            label = key.replace("ROI2_", "")
            ctk.CTkLabel(ctrl, text=f"{label}:", width=28).grid(row=3, column=col*2, padx=2)
            var = ctk.StringVar(value=f"{self._roi[key]:.4f}")
            self._vars2[key] = var
            entry = ctk.CTkEntry(ctrl, textvariable=var, width=72,
                                 state="normal" if self._enable_roi2 else "disabled")
            entry.grid(row=3, column=col*2+1, padx=2)
            entry.bind("<FocusOut>", lambda e, k=key: self._on_entry_change(k, 2))
            entry.bind("<Return>",   lambda e, k=key: self._on_entry_change(k, 2))
            self._roi2_entries.append(entry)

        # Hint
        ctk.CTkLabel(ctrl, text="Kéo bên trong ô để di chuyển  •  Kéo góc dưới-phải để thay đổi kích thước",
                     text_color="gray", font=("", 11)).grid(
            row=4, column=0, columnspan=8, pady=(4, 0))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=6)
        ctk.CTkButton(btn_frame, text="💾 Lưu vào config.py",
                      fg_color="#1a5276", hover_color="#21618c",
                      command=self._save).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Đóng",
                      command=self._close).pack(side="left", padx=8)

    # ----------------------------------------------------------
    def _update_frame(self):
        if not self._running:
            return

        frame = self._camera.get_frame()
        if frame is not None:
            # Resize cho canvas
            disp = cv2.resize(frame, (self._cw, self._ch),
                              interpolation=cv2.INTER_AREA)

            # Vẽ ROI 1
            self._draw_roi_on(disp, "ROI_X", "ROI_Y", "ROI_W", "ROI_H",
                              self.COLOR_ROI1, "Lane 1")

            # Vẽ ROI 2 nếu bật
            if self._roi2_enabled_var.get():
                self._draw_roi_on(disp, "ROI2_X", "ROI2_Y", "ROI2_W", "ROI2_H",
                                  self.COLOR_ROI2, "Lane 2")

            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self._tk_img = ImageTk.PhotoImage(pil)
            self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        if self._running:
            self.after(40, self._update_frame)   # ~25 FPS

    # ----------------------------------------------------------
    def _draw_roi_on(self, frame, xk, yk, wk, hk, color, label):
        cw, ch = self._cw, self._ch
        rx = int(self._roi[xk] * cw)
        ry = int(self._roi[yk] * ch)
        rw = int(self._roi[wk] * cw)
        rh = int(self._roi[hk] * ch)

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)

        # Góc L
        cl = 14
        for (cx, cy) in [(rx, ry), (rx+rw, ry), (rx, ry+rh), (rx+rw, ry+rh)]:
            dx = 1 if cx == rx else -1
            dy = 1 if cy == ry else -1
            cv2.line(frame, (cx, cy), (cx + dx*cl, cy), color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy*cl), color, 3)

        # Handle resize ở góc dưới-phải
        cv2.rectangle(frame,
                      (rx + rw - 10, ry + rh - 10),
                      (rx + rw + 2,  ry + rh + 2),
                      color, -1)

        cv2.putText(frame, label, (rx, max(ry - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # ----------------------------------------------------------
    def _roi_rect_px(self, xk, yk, wk, hk):
        """Trả về (rx, ry, rw, rh) theo pixel canvas."""
        return (
            int(self._roi[xk] * self._cw),
            int(self._roi[yk] * self._ch),
            int(self._roi[wk] * self._cw),
            int(self._roi[hk] * self._ch),
        )

    def _hit_test(self, mx, my):
        """
        Kiểm tra chuột đang ở đâu.
        Trả về (target, mode): target="roi1"|"roi2", mode="resize"|"move"
        """
        RESIZE_MARGIN = 14

        def _check(xk, yk, wk, hk, name):
            rx, ry, rw, rh = self._roi_rect_px(xk, yk, wk, hk)
            # Góc resize dưới-phải
            if (rx + rw - RESIZE_MARGIN <= mx <= rx + rw + RESIZE_MARGIN and
                    ry + rh - RESIZE_MARGIN <= my <= ry + rh + RESIZE_MARGIN):
                return name, "resize"
            # Bên trong ô -> move
            if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                return name, "move"
            return None, None

        t, m = _check("ROI_X", "ROI_Y", "ROI_W", "ROI_H", "roi1")
        if t:
            return t, m

        if self._roi2_enabled_var.get():
            t, m = _check("ROI2_X", "ROI2_Y", "ROI2_W", "ROI2_H", "roi2")
            if t:
                return t, m

        return None, None

    # ----------------------------------------------------------
    def _on_press(self, event):
        self._drag_target, self._drag_mode = self._hit_test(event.x, event.y)
        if self._drag_target:
            self._drag_start = (event.x, event.y)
            self._drag_orig  = dict(self._roi)

    def _on_drag(self, event):
        if not self._drag_target:
            return

        dx = (event.x - self._drag_start[0]) / self._cw
        dy = (event.y - self._drag_start[1]) / self._ch

        if self._drag_target == "roi1":
            xk, yk, wk, hk = "ROI_X", "ROI_Y", "ROI_W", "ROI_H"
        else:
            xk, yk, wk, hk = "ROI2_X", "ROI2_Y", "ROI2_W", "ROI2_H"

        if self._drag_mode == "move":
            nx = max(0.0, min(self._drag_orig[xk] + dx, 1.0 - self._roi[wk]))
            ny = max(0.0, min(self._drag_orig[yk] + dy, 1.0 - self._roi[hk]))
            self._roi[xk] = nx
            self._roi[yk] = ny
        elif self._drag_mode == "resize":
            nw = max(0.05, min(self._drag_orig[wk] + dx, 1.0 - self._roi[xk]))
            nh = max(0.05, min(self._drag_orig[hk] + dy, 1.0 - self._roi[yk]))
            self._roi[wk] = nw
            self._roi[hk] = nh

        # Đồng bộ entry boxes
        vars_map = {**self._vars1, **self._vars2}
        for k in [xk, yk, wk, hk]:
            if k in vars_map:
                vars_map[k].set(f"{self._roi[k]:.4f}")

    def _on_release(self, event):
        self._drag_target = None
        self._drag_mode   = None

    # ----------------------------------------------------------
    def _on_entry_change(self, key, lane):
        """Cập nhật _roi khi người dùng gõ trực tiếp vào entry."""
        try:
            vars_map = self._vars1 if lane == 1 else self._vars2
            val = float(vars_map[key].get())
            val = max(0.0, min(val, 1.0))
            self._roi[key] = val
        except ValueError:
            pass

    def _on_toggle_roi2(self):
        state = "normal" if self._roi2_enabled_var.get() else "disabled"
        for entry in self._roi2_entries:
            entry.configure(state=state)

    # ----------------------------------------------------------
    def _save(self):
        """Ghi giá trị ROI mới vào settings.ini (hoạt động cả khi đóng gói .exe)."""
        from settings_manager import save_roi
        ok = save_roi(self._roi, self._roi2_enabled_var.get())
        if ok:
            self._show_msg("Đã lưu ROI! Có hiệu lực ngay khi khởi động lại app.")
        else:
            self._show_msg("Không lưu được settings.ini.", error=True)

    def _show_msg(self, msg, error=False):
        import tkinter.messagebox as msgbox
        if error:
            msgbox.showerror("ROI Editor", msg, parent=self)
        else:
            msgbox.showinfo("ROI Editor", msg, parent=self)

    # ----------------------------------------------------------
    def _close(self):
        self._running = False
        self.destroy()

    def destroy(self):
        self._running = False
        super().destroy()