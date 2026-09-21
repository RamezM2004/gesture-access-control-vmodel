"""
gesture_camera.py

Raspberry Pi USB-camera gesture comparison module.
Compatible with Python 3.11.9.

Final-system role:
    password_fingers = fetch_password_fingers()
    access_granted = run_gesture_camera(password_fingers)

Finger order:
    [thumb, index, middle, ring, pinky]

1 = raised/open
0 = closed/folded

Requirements inside the Python 3.11.9 venv:
    python -m pip install opencv-python mediapipe numpy

Required model file:
    hand_landmarker.task

Recommended run on Raspberry Pi:
    python gesture_camera.py

If running by SSH without VNC/monitor:
    keep show_window=False, otherwise cv2.imshow may crash because there is no display.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# SETTINGS
# =========================

DEFAULT_CAMERA_INDEX = 0
DEFAULT_MODEL_PATH = "hand_landmarker.task"
DEFAULT_ATTEMPT_TIMEOUT = 20.0
DEFAULT_STABLE_TIME = 0.8
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_FPS = 15


# =========================
# VALIDATION
# =========================

def validate_fingers(fingers: Iterable[int]) -> list[int]:
    """
    Validate and normalize a 5-bit finger password.

    Expected order:
        [thumb, index, middle, ring, pinky]

    Returns:
        list[int] of length 5

    Raises:
        ValueError if invalid
    """
    normalized = [int(x) for x in fingers]

    if len(normalized) != 5:
        raise ValueError(f"Password must contain exactly 5 values, got: {normalized}")

    if any(x not in (0, 1) for x in normalized):
        raise ValueError(f"Password values must be only 0 or 1, got: {normalized}")

    return normalized


# =========================
# CAMERA SETUP
# =========================

def open_usb_camera(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
    fps: int = DEFAULT_FPS,
) -> Optional[cv2.VideoCapture]:
    """
    Open Raspberry Pi USB camera safely using V4L2.

    Returns:
        cv2.VideoCapture object if opened
        None if failed
    """

    # CAP_V4L2 is preferred on Raspberry Pi/Linux for USB cameras.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap.release()
        print(f"ERROR: USB camera could not be opened at index {camera_index}.")
        print("Try:")
        print("  ls /dev/video*")
        print("  python gesture_camera.py")
        print("  python gesture_camera.py --camera 1")
        return None

    # These settings avoid very high resolution/fps that can freeze or overload the Pi.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # MJPG usually works better with USB webcams on Raspberry Pi.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Small buffer reduces lag/freeze from queued old frames.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warm up camera briefly.
    for _ in range(5):
        cap.read()
        time.sleep(0.03)

    print(f"Camera opened successfully at index {camera_index}.")
    return cap


# =========================
# MEDIAPIPE SETUP
# =========================

def create_hand_landmarker(model_path: str = DEFAULT_MODEL_PATH):
    """
    Create MediaPipe HandLandmarker object.
    """
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Put hand_landmarker.task in the same folder as this script."
        )

    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_file)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return HandLandmarker.create_from_options(options)


# =========================
# FINGER DETECTION
# =========================

def get_finger_states(hand_landmarks) -> list[int]:
    """
    Convert MediaPipe hand landmarks to a 5-bit finger array.

    Returns:
        [thumb, index, middle, ring, pinky]

    Note:
        This simple method works best when the palm faces the camera.
    """
    lm = hand_landmarks

    index_open = lm[8].y < lm[6].y
    middle_open = lm[12].y < lm[10].y
    ring_open = lm[16].y < lm[14].y
    pinky_open = lm[20].y < lm[18].y

    # Simple thumb approximation. It is not perfect for every hand rotation,
    # but it is stable enough for the lab prototype.
    thumb_open = abs(lm[4].x - lm[2].x) > 0.08

    return [
        1 if thumb_open else 0,
        1 if index_open else 0,
        1 if middle_open else 0,
        1 if ring_open else 0,
        1 if pinky_open else 0,
    ]


def fingers_to_text(fingers: Optional[list[int]]) -> str:
    if fingers is None:
        return "NO_HAND"
    return "[" + ", ".join(str(x) for x in fingers) + "]"


# =========================
# DRAWING FUNCTIONS
# =========================

def draw_hand(frame, hand_landmarks) -> None:
    h, w, _ = frame.shape

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    for landmark in hand_landmarks:
        cx = int(landmark.x * w)
        cy = int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    for start, end in connections:
        x1 = int(hand_landmarks[start].x * w)
        y1 = int(hand_landmarks[start].y * h)
        x2 = int(hand_landmarks[end].x * w)
        y2 = int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.7, thickness=2) -> None:
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


# =========================
# MAIN CAMERA COMPARISON FUNCTION
# =========================

def run_gesture_camera(
    password_fingers: Iterable[int],
    camera_index: int = DEFAULT_CAMERA_INDEX,
    model_path: str = DEFAULT_MODEL_PATH,
    attempt_timeout: float = DEFAULT_ATTEMPT_TIMEOUT,
    stable_time: float = DEFAULT_STABLE_TIME,
    show_window: bool = False,
    print_every_seconds: float = 0.5,
) -> bool:
    """
    Open USB camera, detect hand, compare detected fingers with password_fingers.

    Args:
        password_fingers:
            Required 5-bit finger password from password_client.py.
            Example: [0, 1, 1, 0, 0]

        camera_index:
            USB camera index, usually 0 on Raspberry Pi.

        model_path:
            Path to hand_landmarker.task.

        attempt_timeout:
            Maximum seconds to keep camera open for this attempt.
            Prevents Raspberry Pi from freezing or camera staying open forever.

        stable_time:
            User must hold the correct finger array for this many seconds.

        show_window:
            False is safest for SSH/headless.
            True is okay if using Raspberry Pi desktop/VNC/monitor.

    Returns:
        True  -> access granted
        False -> wrong gesture, timeout, camera/model error, or quit
    """

    try:
        required = validate_fingers(password_fingers)
    except ValueError as exc:
        print(f"ERROR: Invalid password_fingers: {exc}")
        return False

    print("Required finger password:", required)

    cap = open_usb_camera(camera_index=camera_index)
    if cap is None:
        return False

    start_time = time.time()
    last_print_time = 0.0
    last_detected_fingers: Optional[list[int]] = None
    stable_start_time: Optional[float] = None

    try:
        with create_hand_landmarker(model_path=model_path) as landmarker:
            while True:
                current_time = time.time()

                if current_time - start_time >= attempt_timeout:
                    print("ACCESS DENIED: gesture attempt timed out.")
                    return False

                ret, frame = cap.read()
                if not ret or frame is None:
                    print("WARNING: Could not read frame. Retrying...")
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = int((current_time - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                detected_fingers: Optional[list[int]] = None

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]
                    detected_fingers = get_finger_states(hand_landmarks)

                    if show_window:
                        draw_hand(frame, hand_landmarks)

                # Stability check: same detected array must stay stable.
                if detected_fingers == last_detected_fingers:
                    if stable_start_time is None:
                        stable_start_time = current_time
                else:
                    stable_start_time = current_time
                    last_detected_fingers = detected_fingers

                stable_duration = current_time - stable_start_time if stable_start_time else 0.0

                # Print terminal status without flooding the Pi.
                if current_time - last_print_time >= print_every_seconds:
                    print(
                        "Detected:", fingers_to_text(detected_fingers),
                        "| Required:", fingers_to_text(required),
                        f"| Stable: {stable_duration:.1f}s"
                    )
                    last_print_time = current_time

                if detected_fingers == required and stable_duration >= stable_time:
                    print("ACCESS GRANTED: correct finger password detected.")
                    return True

                # Optional visual window. Do not use over plain SSH.
                if show_window:
                    remaining = max(0, int(attempt_timeout - (current_time - start_time)))
                    draw_text(frame, f"Detected: {fingers_to_text(detected_fingers)}", (30, 50), (0, 255, 0))
                    draw_text(frame, f"Required: {fingers_to_text(required)}", (30, 90), (255, 255, 255))
                    draw_text(frame, f"Stable: {stable_duration:.1f}s", (30, 130), (255, 255, 0))
                    draw_text(frame, f"Timeout: {remaining}s", (30, 170), (255, 255, 0))
                    draw_text(frame, "Press Q to quit", (30, frame.shape[0] - 30), (200, 200, 200))

                    cv2.imshow("Gesture Camera", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("Gesture attempt cancelled by user.")
                        return False

                # Tiny sleep prevents CPU from maxing out on the Pi.
                time.sleep(0.005)

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return False
    except KeyboardInterrupt:
        print("Gesture camera stopped by keyboard interrupt.")
        return False
    except Exception as exc:
        print(f"ERROR during gesture camera run: {exc}")
        return False
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        print("Camera released.")


# =========================
# DIRECT TEST
# =========================

if __name__ == "__main__":
    # Test password: PEACE = [thumb, index, middle, ring, pinky]
    test_password = [0, 1, 1, 0, 0]

    result = run_gesture_camera(
        password_fingers=test_password,
        camera_index=0,
        model_path="hand_landmarker.task",
        attempt_timeout=20.0,
        stable_time=0.8,
        show_window=False,  # keep False for SSH. Use True only in VNC/desktop.
    )

    print("Final gesture result:", result)
