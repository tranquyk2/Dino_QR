# ============================================================
#  arduino_sent.py — Giao tiếp với Arduino qua Serial
#  (thay thế PLC_sent.py / Modbus)
#  Hỗ trợ tự động dò cổng theo VID/PID + tự reconnect
# ============================================================

import logging
import threading
import time

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)


# --- VID/PID nhận diện Arduino ---
# CH340 (clone Mega/Uno dùng chip CH340)
CH340_VID = 0x1A86
CH340_PID = 0x7523

# Arduino Mega 2560 chính hãng (ATmega16U2)
MEGA_VID = 0x2341
MEGA_PID = 0x0042


def list_serial_ports() -> list[str]:
    """Liệt kê các cổng COM hiện có trên máy."""
    return [p.device for p in serial.tools.list_ports.comports()]


def find_arduino_port() -> str | None:
    """
    Tự động tìm cổng COM của Arduino dựa vào VID/PID.
    Ưu tiên Arduino Mega 2560 chính hãng, sau đó tới board dùng CH340.
    Trả về None nếu không tìm thấy.
    """
    ports = list(serial.tools.list_ports.comports())

    # Ưu tiên Mega chính hãng
    for p in ports:
        if p.vid == MEGA_VID and p.pid == MEGA_PID:
            return p.device

    # Sau đó tới board dùng CH340 (clone phổ biến)
    for p in ports:
        if p.vid == CH340_VID and p.pid == CH340_PID:
            return p.device

    return None


# --- Cấu hình mặc định ---
ARDUINO_PORT     = "COM3"   # Cổng dự phòng nếu không auto-detect được
ARDUINO_BAUDRATE = 9600
ARDUINO_TIMEOUT  = 0.5


class ArduinoConnection:
    """
    Quản lý kết nối Serial tới Arduino, gửi lệnh NG/OK.
    Hỗ trợ tự động dò cổng theo VID/PID và tự động reconnect khi mất kết nối.
    """

    def __init__(self, port: str | None = None, baudrate: int = ARDUINO_BAUDRATE,
                 auto_detect: bool = True, auto_reconnect: bool = True):
        self.auto_detect = auto_detect
        self.baudrate = baudrate
        self.ser: serial.Serial | None = None
        self._lock = threading.Lock()

        if port:
            self.port = port
        elif auto_detect:
            self.port = find_arduino_port() or ARDUINO_PORT
        else:
            self.port = ARDUINO_PORT

        self._connect()

        # Auto-reconnect monitor
        self._monitor_running = False
        self._monitor_thread: threading.Thread | None = None
        if auto_reconnect:
            self.start_auto_reconnect()

    # ----------------------------------------------------------
    def _connect(self):
        try:
            self.ser = serial.Serial(
                self.port, self.baudrate, timeout=ARDUINO_TIMEOUT
            )
            # Arduino reset khi mở cổng serial -> chờ board khởi động
            time.sleep(2.0)
            logger.info(f"Đã kết nối Arduino trên {self.port} @ {self.baudrate} baud")
        except Exception as e:
            logger.error(f"Lỗi khi mở kết nối Arduino: {e}")
            self.ser = None

    # ----------------------------------------------------------
    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    # ----------------------------------------------------------
    def reconnect(self, port: str | None = None, baudrate: int = ARDUINO_BAUDRATE) -> bool:
        """Đóng kết nối hiện tại và kết nối lại với thông số mới.
        Nếu port=None, tự dò lại theo VID/PID."""
        self._close_serial()
        if port:
            self.port = port
        elif self.auto_detect:
            self.port = find_arduino_port() or self.port
        self.baudrate = baudrate
        self._connect()
        return self.is_connected()

    # ----------------------------------------------------------
    def start_auto_reconnect(self, interval: float = 2.0):
        """Bắt đầu thread tự động dò & kết nối lại khi mất kết nối Arduino."""
        if self._monitor_running:
            return
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._auto_reconnect_loop, args=(interval,),
            daemon=True, name="ArduinoAutoReconnect"
        )
        self._monitor_thread.start()

    def stop_auto_reconnect(self):
        self._monitor_running = False

    def _auto_reconnect_loop(self, interval: float):
        while self._monitor_running:
            try:
                if not self.is_connected():
                    new_port = None
                    if self.auto_detect:
                        new_port = find_arduino_port()
                    target_port = new_port or self.port
                    if target_port:
                        logger.info(f"Đang thử kết nối lại Arduino trên {target_port}...")
                        self.port = target_port
                        self._connect()
            except Exception as e:
                logger.error(f"Lỗi trong auto-reconnect: {e}")
            time.sleep(interval)

    # ----------------------------------------------------------
    def _send(self, command: str) -> bool:
        """Gửi 1 lệnh dạng text, kết thúc bằng '\\n'.
        Không tự reconnect đồng bộ (auto-reconnect thread sẽ xử lý)."""
        if not self.is_connected():
            return False
        try:
            with self._lock:
                self.ser.write((command.strip() + "\n").encode("utf-8"))
            return True
        except Exception as e:
            logger.error(f"Lỗi khi gửi lệnh '{command}': {e}")
            self._close_serial()
            return False

    # ----------------------------------------------------------
    def send_ng(self) -> bool:
        """Báo NG -> Arduino dừng motor."""
        return self._send("NG")

    def send_ok(self) -> bool:
        """Báo OK -> xóa cờ NG (không tự chạy lại motor)."""
        return self._send("OK")

    def request_status(self) -> str | None:
        """Gửi lệnh STATUS và đọc phản hồi (nếu có)."""
        if not self._send("STATUS"):
            return None
        try:
            with self._lock:
                line = self.ser.readline().decode("utf-8", errors="replace").strip()
            return line if line else None
        except Exception as e:
            logger.error(f"Lỗi khi đọc trạng thái Arduino: {e}")
            self._close_serial()
            return None

    # ----------------------------------------------------------
    def _close_serial(self):
        """Đóng cổng serial hiện tại (không dừng monitor thread)."""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception as e:
            logger.error(f"Lỗi khi đóng cổng Arduino: {e}")
        self.ser = None

    def close(self):
        """Dừng auto-reconnect và đóng kết nối hoàn toàn."""
        self.stop_auto_reconnect()
        self._close_serial()


# --- Test độc lập ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Đang dò cổng Arduino theo VID/PID...")
    found = find_arduino_port()
    print(f"Cổng tìm được: {found}")

    arduino = ArduinoConnection()
    if arduino.is_connected():
        print(f"Đã kết nối: {arduino.port}")
        print("Gửi NG...")
        print("OK" if arduino.send_ng() else "Thất bại")
        time.sleep(0.5)

        print("Hỏi trạng thái...")
        print(arduino.request_status())

        time.sleep(0.5)
        print("Gửi OK...")
        print("OK" if arduino.send_ok() else "Thất bại")
    else:
        print("Không kết nối được Arduino. Kiểm tra cổng COM / cáp.")

    arduino.close()