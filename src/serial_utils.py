"""
Serial Utilities — Shared micro:bit port detection
====================================================
Centralized utility for detecting BBC micro:bit serial ports.
Used by capture.py, server.py, and trainer.py to avoid code duplication.
"""

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def get_microbit_port():
    """Auto-detect micro:bit serial port (Windows, Mac, Linux).

    Returns the device path string if found, or None if no micro:bit
    is detected.
    """
    if not HAS_SERIAL:
        return None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid == 0x0D28 and p.pid == 0x0204:
            return p.device
        desc = p.description.lower()
        if "micro:bit" in desc or "mbed" in desc:
            return p.device
        if "usbmodem" in p.device.lower():
            return p.device
    return None
