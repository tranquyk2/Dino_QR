# ============================================================
#  config.py — Cấu hình toàn cục cho QR Defect Detector
# ============================================================

# --- Camera ---
CAMERA_INDEX   = 0
CAMERA_BACKEND = "DSHOW"   # "DSHOW" | "MSMF" | "AUTO"
CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 960
CAMERA_FPS     = 30

# --- QR Detection ---
ENABLE_ROI = True
ROI_X = 0.0041
ROI_Y = 0.0087
ROI_W = 0.5033
ROI_H = 0.5905

# --- ROI 2 ---
ENABLE_ROI2 = True
ROI2_X = 0.5073
ROI2_Y = 0.0132
ROI2_W = 0.4927
ROI2_H = 0.5850

# --- Presence detection (QR rách / mất finder pattern) ---
ENABLE_PRESENCE_DETECTION  = False
PRESENCE_STD_THRESHOLD     = 30.0
PRESENCE_NG_FRAME_THRESHOLD = 10
DEBUG_PRINT_ROI_STD        = False

# --- Timing ---
RESULT_HOLD_SECONDS = 2.0
NG_FRAME_THRESHOLD  = 7      # Khuyến nghị 3 (giảm từ 8 → nhanh hơn ~200ms)

# --- Preprocessing ---
ENABLE_CLAHE = True
CLAHE_CLIP   = 2.0
CLAHE_GRID   = (8, 8)

# --- Paths ---
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục lưu ảnh NG  (có thể thay đổi từ dialog Cài đặt → ghi vào file này)
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")

# Đường dẫn file CSV log OK  (có thể thay đổi từ dialog Cài đặt)
CSV_PATH = os.path.join(BASE_DIR, "logs", "ok_log.csv")

# --- Tương thích ngược (các module cũ import LOG_DIR / CSV_DIR) ---
LOG_DIR = os.path.join(BASE_DIR, "logs")
CSV_DIR = LOG_DIR

# Tạo thư mục mặc định nếu chưa có
os.makedirs(CAPTURE_DIR,              exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
os.makedirs(LOG_DIR,                  exist_ok=True)

# --- Logging / Capture ---
ENABLE_NG_CAPTURE = True    # Bật/tắt lưu ảnh NG
ENABLE_OK_CSV_LOG = True    # Bật/tắt ghi log OK vào CSV

# --- GUI ---
APP_TITLE = "QR Code Defect Detector — Dino-Lite"
WINDOW_W  = 1280
WINDOW_H  = 760
THEME     = "dark"   # "dark" | "light"


# ── Đọc đè từ settings.ini (nếu có) ─────────────────────────
# Thực hiện SAU khi đã khai báo CAPTURE_DIR, CSV_PATH, ROI_* ở trên
# để settings.ini có thể ghi đè giá trị mặc định.
try:
    from settings_manager import load_roi, load_paths, load_logging_flags

    _roi = load_roi()
    ROI_X       = _roi["ROI_X"]
    ROI_Y       = _roi["ROI_Y"]
    ROI_W       = _roi["ROI_W"]
    ROI_H       = _roi["ROI_H"]
    ENABLE_ROI2 = _roi["ENABLE_ROI2"]
    ROI2_X      = _roi["ROI2_X"]
    ROI2_Y      = _roi["ROI2_Y"]
    ROI2_W      = _roi["ROI2_W"]
    ROI2_H      = _roi["ROI2_H"]

    _cap, _csv = load_paths(CAPTURE_DIR, CSV_PATH)
    CAPTURE_DIR = _cap
    CSV_PATH    = _csv

    _ng, _ok = load_logging_flags()
    ENABLE_NG_CAPTURE = _ng
    ENABLE_OK_CSV_LOG = _ok

except Exception:
    pass  # Nếu settings_manager chưa có, dùng giá trị mặc định ở trên


# ── Hàm lưu paths (giữ tương thích ngược với main.py) ────────
def save_paths_to_config(capture_dir: str, csv_path: str) -> bool:
    """Lưu đường dẫn vào settings.ini (không còn ghi vào config.py)."""
    try:
        from settings_manager import save_paths
        return save_paths(capture_dir, csv_path)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Không lưu được paths: {e}")
        return False