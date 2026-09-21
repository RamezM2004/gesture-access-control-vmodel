from gpiozero import Button
import time


# =========================
# GPIO SETTINGS
# =========================

MYRIO_TRIGGER_PIN = 26   # GPIO26 = physical pin 37

# pull_up=False:
# LOW  = 0V
# HIGH = trigger from myRIO
myrio_trigger = Button(
    MYRIO_TRIGGER_PIN,
    pull_up=False,
    bounce_time=0.2
)


def wait_for_ir_trigger():
    """
    Wait until myRIO sends HIGH signal to GPIO26.

    Returns:
        "ON" when trigger is received
    """

    print("Waiting for myRIO trigger on GPIO26 / physical pin 37...")

    while True:
        if myrio_trigger.is_pressed:
            print("myRIO trigger received.")

            # Wait until signal goes LOW again to avoid repeated triggers
            while myrio_trigger.is_pressed:
                time.sleep(0.05)

            return "ON"

        time.sleep(0.05)


# Test this file directly
if __name__ == "__main__":
    status = wait_for_ir_trigger()
    print("Returned status:", status)