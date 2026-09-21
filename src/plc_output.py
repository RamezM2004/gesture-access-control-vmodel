from pymodbus.client import ModbusTcpClient
import time


# =========================
# PLC SETTINGS - DELTA PLC
# =========================

PLC_IP = "192.168.1.5"
PLC_PORT = 502

# Delta PLC M address mapping:
# M0 is often 2048
# M1 is often 2049
M1_COIL_ADDRESS = 2049


def turn_m1_on():
    """
    Send command to Delta PLC to turn M1 ON using Modbus TCP.
    """

    try:
        client = ModbusTcpClient(PLC_IP, port=PLC_PORT, timeout=3)

        if not client.connect():
            print("PLC ERROR: Could not connect to PLC at", PLC_IP)
            return False

        result = client.write_coil(M1_COIL_ADDRESS, True)
        client.close()

        if result.isError():
            print("PLC ERROR: Failed to write M1 ON")
            print(result)
            return False

        print("PLC COMMAND SENT: M1 ON")
        return True

    except Exception as e:
        print("PLC ERROR while turning M1 ON:", e)
        return False


def turn_m1_off():
    """
    Send command to Delta PLC to turn M1 OFF using Modbus TCP.
    """

    try:
        client = ModbusTcpClient(PLC_IP, port=PLC_PORT, timeout=3)

        if not client.connect():
            print("PLC ERROR: Could not connect to PLC at", PLC_IP)
            return False

        result = client.write_coil(M1_COIL_ADDRESS, False)
        client.close()

        if result.isError():
            print("PLC ERROR: Failed to write M1 OFF")
            print(result)
            return False

        print("PLC COMMAND SENT: M1 OFF")
        return True

    except Exception as e:
        print("PLC ERROR while turning M1 OFF:", e)
        return False


def run_plc_output(pulse_seconds=2.0):
    """
    Final function for main.py.
    Turns M1 ON, waits, then turns M1 OFF.
    """

    print("Sending PLC output pulse...")

    ok_on = turn_m1_on()

    if not ok_on:
        return False

    time.sleep(pulse_seconds)

    ok_off = turn_m1_off()

    if not ok_off:
        return False

    print("PLC output pulse completed.")
    return True


if __name__ == "__main__":
    run_plc_output(pulse_seconds=2.0)