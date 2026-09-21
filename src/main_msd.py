from get_ir2 import wait_for_ir_trigger
from password_client import fetch_password_fingers
from gesture_camera import run_gesture_camera
from plc_output import run_plc_output


while True:
    status = wait_for_ir_trigger()

    if status == "ON":
        password_fingers = fetch_password_fingers()

        if password_fingers is None:
            print("Password fetch failed. Waiting again.")
            continue

        access_granted = run_gesture_camera(password_fingers)

        if access_granted:
            print("Correct gesture. Sending command to PLC.")

            plc_ok = run_plc_output(pulse_seconds=2.0)

            if plc_ok:
                print("PLC output completed successfully.")
            else:
                print("PLC output failed.")

        else:
            print("Wrong gesture. PLC will not run.")