"""
main_system.py

Integrated Raspberry Pi main program for:
    myRIO IR trigger  ->  Raspberry Pi gesture password  ->  PLC Modbus TCP relay command

System flow:
    1. myRIO reads IR/distance sensor.
    2. If user is close, myRIO sends USER_DETECTED\n over UART.
    3. Raspberry Pi receives USER_DETECTED.
    4. Raspberry Pi opens camera and runs one-gesture password check.
    5. If gesture is correct, Raspberry Pi sends Modbus TCP command to PLC.
    6. PLC ladder logic turns ON relay output Y0.

Required files in the same folder:
    main_system.py
    gesture_password.py
    uart_wait_trigger.py
    plc_modbus_tcp.py
    hand_landmarker.task

Install dependencies on Raspberry Pi:
    python3 -m pip install opencv-python mediapipe pyserial pymodbus

Run:
    python3 main_system.py

Before lab testing:
    - Replace UART port if needed: /dev/ttyUSB0, /dev/ttyACM0, or /dev/serial0
    - Replace PLC IP address with the real PLC IP
    - Replace Modbus coil address with the PLC coil mapped in ladder logic
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from gesture_password import run_gesture_password
from plc_output import send_access_granted_to_plc
from uart_wait_trigger import wait_for_user_detected


# ============================================================
# MAIN CONFIGURATION — CHANGE PLACEHOLDERS IN THE LAB
# ============================================================

@dataclass
class SystemConfig:
    """All important settings for the integrated system."""

    # -------------------------
    # Test mode switches
    # -------------------------
    # True  = wait for real myRIO UART trigger
    # False = skip UART and start gesture immediately, useful for laptop testing
    use_uart_trigger: bool = True

    # True  = send real Modbus TCP command to PLC
    # False = only print what would be sent, useful before connecting PLC
    use_plc_output: bool = True

    # True  = keep running after one attempt
    # False = run one trigger/gesture/PLC cycle only, useful for debugging
    continuous_mode: bool = True

    # Delay between completed cycles
    cycle_delay_seconds: float = 1.0

    # -------------------------
    # UART / myRIO settings
    # -------------------------
    uart_port: str = "/dev/ttyUSB0"      # TODO: replace if adapter appears differently
    uart_baudrate: int = 9600            # Must match LabVIEW/myRIO UART setting
    uart_trigger_message: str = "USER_DETECTED"
    uart_wait_timeout: float | None = None   # None = wait forever

    # -------------------------
    # Gesture recognition settings
    # -------------------------
    camera_index: int = 0
    model_path: str = "hand_landmarker.task"
    gesture_attempt_timeout: float = 20.0
    show_camera_window: bool = True

    # -------------------------
    # PLC / Modbus TCP settings
    # -------------------------
    plc_ip: str = "192.168.1.100"        # TODO: replace with real PLC IP in lab
    plc_port: int = 502                  # Standard Modbus TCP port
    plc_slave_id: int = 1                # Try 1 first; if needed try 0
    plc_access_coil: int = 0             # TODO: replace with mapped PLC coil address
    plc_pulse_seconds: float = 2.0       # Relay command pulse duration


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_required_files(config: SystemConfig) -> bool:
    """Check files needed by the integrated system."""

    ok = True

    if not Path(config.model_path).exists():
        print(f"ERROR: MediaPipe model file not found: {config.model_path}")
        print("Put hand_landmarker.task in the same folder as main_system.py, or update model_path.")
        ok = False

    return ok


def wait_for_trigger(config: SystemConfig) -> bool:
    """Wait for myRIO trigger, or bypass it during testing."""

    if not config.use_uart_trigger:
        print("UART trigger bypassed for testing. Starting gesture recognition directly.")
        return True

    print("Waiting for myRIO UART trigger...")
    return wait_for_user_detected(
        port=config.uart_port,
        baudrate=config.uart_baudrate,
        timeout=config.uart_wait_timeout,
        trigger_message=config.uart_trigger_message,
        verbose=True,
    )


def run_gesture_attempt(config: SystemConfig) -> bool:
    """Run the one-gesture password check."""

    print("Starting gesture password attempt...")

    access_granted = run_gesture_password(
        camera_index=config.camera_index,
        model_path=config.model_path,
        attempt_timeout=config.gesture_attempt_timeout,
        show_window=config.show_camera_window,
    )

    if access_granted:
        print("Gesture result: ACCESS GRANTED")
    else:
        print("Gesture result: ACCESS DENIED")

    return access_granted


def activate_plc(config: SystemConfig) -> bool:
    """Send access-granted command to PLC, or bypass it during testing."""

    if not config.use_plc_output:
        print("PLC output bypassed for testing. Would send ACCESS GRANTED command to PLC now.")
        return True

    print("Sending ACCESS GRANTED command to PLC over Modbus TCP...")

    plc_ok = send_access_granted_to_plc(
        ip=config.plc_ip,
        port=config.plc_port,
        slave_id=config.plc_slave_id,
        coil_address=config.plc_access_coil,
        pulse_seconds=config.plc_pulse_seconds,
    )

    if plc_ok:
        print("PLC command sent successfully.")
    else:
        print("PLC command failed. Check PLC IP, port, slave ID, Modbus mapping, and ladder logic.")

    return plc_ok


def run_one_cycle(config: SystemConfig) -> bool:
    """
    Run one complete system cycle.

    Returns:
        True  -> cycle completed and PLC command was sent or bypassed successfully
        False -> trigger failed, gesture denied, or PLC command failed
    """

    print("\n============================================================")
    print("New system cycle")
    print("============================================================")

    triggered = wait_for_trigger(config)
    if not triggered:
        print("No valid trigger received. Cycle stopped.")
        return False

    access_granted = run_gesture_attempt(config)
    if not access_granted:
        print("Access denied. PLC will not be activated.")
        return False

    plc_ok = activate_plc(config)
    return plc_ok


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    config = SystemConfig(
        # During laptop testing, you can set these two to False:
        use_uart_trigger=True,
        use_plc_output=True,

        # For first integration test, set continuous_mode=False.
        continuous_mode=True,

        # UART placeholder
        uart_port="/dev/ttyUSB0",
        uart_baudrate=9600,
        uart_trigger_message="USER_DETECTED",

        # Gesture placeholder
        camera_index=0,
        model_path="hand_landmarker.task",
        gesture_attempt_timeout=20.0,
        show_camera_window=True,

        # PLC placeholder
        plc_ip="192.168.1.100",
        plc_port=502,
        plc_slave_id=1,
        plc_access_coil=0,
        plc_pulse_seconds=2.0,
    )

    print("Integrated myRIO + Raspberry Pi + PLC system started.")
    print("Press Ctrl+C in the terminal to stop.")

    if not check_required_files(config):
        return

    try:
        if config.continuous_mode:
            while True:
                run_one_cycle(config)
                time.sleep(config.cycle_delay_seconds)
        else:
            run_one_cycle(config)

    except KeyboardInterrupt:
        print("\nSystem stopped by user.")


if __name__ == "__main__":
    main()
