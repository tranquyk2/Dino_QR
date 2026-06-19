# ============================================================
#  settings_manager.py — Lưu/đọc cài đặt runtime vào settings.ini
#  Đặt cạnh file .exe (hoặc cạnh main.py khi chạy từ source).
#  Thay thế hoàn toàn việc ghi đè config.py.
# ============================================================

import os
import sys
import configparser
import logging

log = logging.getLogger(__name__)


def _get_settings_path() -> str:
    """
    Trả về đường dẫn tuyệt đối của settings.ini.
    - Khi chạy từ .exe (PyInstaller): cạnh file .exe
    - Khi chạy từ source: cạnh main.py
    """
    if getattr(sys, "frozen", False):
        # Đang chạy từ .exe
        base = os.path.dirname(sys.executable)
    else:
        # Đang chạy từ source
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "settings.ini")


SETTINGS_PATH = _get_settings_path()

# Giá trị mặc định — lấy từ config.py lần đầu, sau đó settings.ini ghi đè
_DEFAULTS = {
    "roi": {
        "roi_x":       "0.0041",
        "roi_y":       "0.0087",
        "roi_w":       "0.5033",
        "roi_h":       "0.5905",
        "enable_roi2": "True",
        "roi2_x":      "0.5073",
        "roi2_y":      "0.0132",
        "roi2_w":      "0.4927",
        "roi2_h":      "0.5850",
    },
    "paths": {
        "capture_dir": "",   # Sẽ điền theo BASE_DIR lúc load
        "csv_path":    "",
    },
    "logging": {
        "enable_ng_capture": "True",
        "enable_ok_csv_log": "True",
    },
}


def _load_raw() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    # Nạp default trước
    for section, kvs in _DEFAULTS.items():
        cfg[section] = kvs
    # Đọc đè từ file (nếu có)
    cfg.read(SETTINGS_PATH, encoding="utf-8")
    return cfg


def _save_raw(cfg: configparser.ConfigParser):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            cfg.write(f)
        log.info(f"settings.ini đã lưu → {SETTINGS_PATH}")
        return True
    except Exception as e:
        log.error(f"Không lưu được settings.ini: {e}")
        return False


# ── API công khai ──────────────────────────────────────────────

def load_roi() -> dict:
    """Trả về dict ROI từ settings.ini (hoặc default)."""
    cfg = _load_raw()
    s = cfg["roi"]
    return {
        "ROI_X":       float(s.get("roi_x",       _DEFAULTS["roi"]["roi_x"])),
        "ROI_Y":       float(s.get("roi_y",       _DEFAULTS["roi"]["roi_y"])),
        "ROI_W":       float(s.get("roi_w",       _DEFAULTS["roi"]["roi_w"])),
        "ROI_H":       float(s.get("roi_h",       _DEFAULTS["roi"]["roi_h"])),
        "ENABLE_ROI2": s.get("enable_roi2", _DEFAULTS["roi"]["enable_roi2"]).lower() == "true",
        "ROI2_X":      float(s.get("roi2_x",      _DEFAULTS["roi"]["roi2_x"])),
        "ROI2_Y":      float(s.get("roi2_y",      _DEFAULTS["roi"]["roi2_y"])),
        "ROI2_W":      float(s.get("roi2_w",      _DEFAULTS["roi"]["roi2_w"])),
        "ROI2_H":      float(s.get("roi2_h",      _DEFAULTS["roi"]["roi2_h"])),
    }


def save_roi(roi: dict, enable_roi2: bool) -> bool:
    """Lưu giá trị ROI vào settings.ini."""
    cfg = _load_raw()
    if "roi" not in cfg:
        cfg["roi"] = {}
    cfg["roi"]["roi_x"]       = f"{roi['ROI_X']:.4f}"
    cfg["roi"]["roi_y"]       = f"{roi['ROI_Y']:.4f}"
    cfg["roi"]["roi_w"]       = f"{roi['ROI_W']:.4f}"
    cfg["roi"]["roi_h"]       = f"{roi['ROI_H']:.4f}"
    cfg["roi"]["enable_roi2"] = str(enable_roi2)
    cfg["roi"]["roi2_x"]      = f"{roi['ROI2_X']:.4f}"
    cfg["roi"]["roi2_y"]      = f"{roi['ROI2_Y']:.4f}"
    cfg["roi"]["roi2_w"]      = f"{roi['ROI2_W']:.4f}"
    cfg["roi"]["roi2_h"]      = f"{roi['ROI2_H']:.4f}"
    return _save_raw(cfg)


def load_paths(default_capture: str, default_csv: str) -> tuple[str, str]:
    """
    Trả về (capture_dir, csv_path) từ settings.ini.
    Nếu chưa có trong file thì dùng default truyền vào.
    """
    cfg = _load_raw()
    capture = cfg["paths"].get("capture_dir", "").strip() or default_capture
    csv     = cfg["paths"].get("csv_path",    "").strip() or default_csv
    return capture, csv


def save_paths(capture_dir: str, csv_path: str) -> bool:
    """Lưu đường dẫn captures và CSV vào settings.ini."""
    cfg = _load_raw()
    if "paths" not in cfg:
        cfg["paths"] = {}
    cfg["paths"]["capture_dir"] = capture_dir
    cfg["paths"]["csv_path"]    = csv_path
    return _save_raw(cfg)


def load_logging_flags() -> tuple[bool, bool]:
    """Trả về (enable_ng_capture, enable_ok_csv_log)."""
    cfg = _load_raw()
    ng = cfg["logging"].get("enable_ng_capture", "True").lower() == "true"
    ok = cfg["logging"].get("enable_ok_csv_log", "True").lower() == "true"
    return ng, ok


def save_logging_flags(enable_ng: bool, enable_ok: bool) -> bool:
    cfg = _load_raw()
    if "logging" not in cfg:
        cfg["logging"] = {}
    cfg["logging"]["enable_ng_capture"] = str(enable_ng)
    cfg["logging"]["enable_ok_csv_log"] = str(enable_ok)
    return _save_raw(cfg)