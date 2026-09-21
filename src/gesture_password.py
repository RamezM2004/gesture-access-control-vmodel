import json
import os
import sys
import time
import logging
import math
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp  # type: ignore[import-untyped]

# -----------------------------------------------------------------------------
# CONFIGURATION -- edit these for your deployment
# -----------------------------------------------------------------------------

CAMERA_SOURCE:           int   = 0
DISPLAY_WINDOW:          bool  = True        # False for headless RPi production

# Remote endpoint -- PLACEHOLDER: replace with actual device IP and port
REMOTE_IP:               str   = "PLACEHOLDER_IP"   # e.g. "192.168.1.100"
REMOTE_PORT:             int   = 8080
REMOTE_ENDPOINT:         str   = "http://{}:{}/gesture".format(REMOTE_IP, REMOTE_PORT)
REMOTE_FETCH_TIMEOUT:    float = 2.0         # seconds

# Expected JSON response:  {"fingers": [0, 1, 1, 0, 0]}
# Also accepts plain CSV:  0,1,1,0,0

RESULTS_FILE:            str   = "gesture_results.json"

# Test target password -- used when the remote endpoint is unreachable.
# Change these to whatever gesture combination you want to test against.
# [thumb, index, middle, ring, pinky]  0=closed  1=extended
TEST_TARGET_1: list = [1, 0, 0, 0, 1]   # left  hand target
TEST_TARGET_2: list = [0, 1, 0, 0, 1]   # right hand target

GESTURE_HOLD_SECONDS:    float = 1.5         # hold duration to confirm a gesture
DEBOUNCE_WINDOW:         int   = 7           # per-finger rolling-window frames
MIN_DETECT_CONF:         float = 0.7
MIN_TRACK_CONF:          float = 0.6
FINGER_ANGLE_THRESHOLD:  float = 160.0
THUMB_ANGLE_THRESHOLD:   float = 150.0

# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("GesturePassword")


# -- Landmark indices ---------------------------------------------------------

@dataclass(frozen=True)
class _LM:
    THUMB_CMC:  int = 1
    THUMB_MCP:  int = 2
    THUMB_IP:   int = 3
    INDEX_PIP:  int = 6
    INDEX_TIP:  int = 8
    MIDDLE_PIP: int = 10
    MIDDLE_TIP: int = 12
    RING_PIP:   int = 14
    RING_TIP:   int = 16
    PINKY_PIP:  int = 18
    PINKY_TIP:  int = 20

LM = _LM()

_FINGER_PAIRS: Tuple[Tuple[int, int], ...] = (
    (LM.INDEX_TIP,  LM.INDEX_PIP),
    (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
    (LM.RING_TIP,   LM.RING_PIP),
    (LM.PINKY_TIP,  LM.PINKY_PIP),
)


# -- Helpers ------------------------------------------------------------------

def _angle(a, b, c) -> float:
    """Angle ABC in degrees for three 2-D landmarks with .x / .y attributes."""
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if mag == 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (bax * bcx + bay * bcy) / mag))))


def _fmt(g1: List[int], g2: List[int]) -> str:
    return "[{}] [{}]".format(" ".join(map(str, g1)), " ".join(map(str, g2)))


# -- Core components ----------------------------------------------------------

class FingerStateDetector:
    """
    Extracts per-finger binary extension state from MediaPipe landmarks.

    Returns [thumb, index, middle, ring, pinky]  (0=closed, 1=extended).
    Angle-based detection is scale-invariant across camera distances.
    """

    def detect(self, landmarks) -> List[int]:
        lm = landmarks  # Tasks API: List[NormalizedLandmark] passed directly
        states: List[int] = [
            1 if _angle(lm[LM.THUMB_CMC], lm[LM.THUMB_MCP], lm[LM.THUMB_IP])
                 >= THUMB_ANGLE_THRESHOLD else 0
        ]
        for tip, pip in _FINGER_PAIRS:
            states.append(
                1 if _angle(lm[pip - 1], lm[pip], lm[tip])
                     >= FINGER_ANGLE_THRESHOLD else 0
            )
        return states  # length always 5


class FingerDebounceFilter:
    """Per-finger majority-vote rolling window to suppress single-frame noise."""

    def __init__(self, window: int = DEBOUNCE_WINDOW) -> None:
        self._bufs = [deque(maxlen=window) for _ in range(5)]

    def push(self, states: List[int]) -> List[int]:
        out = []
        for i, s in enumerate(states):
            self._bufs[i].append(s)
            buf = self._bufs[i]
            out.append(1 if sum(buf) > len(buf) // 2 else 0)
        return out

    def reset(self) -> None:
        for b in self._bufs:
            b.clear()


class RemoteGestureReader:
    """
    Reads a 5-element finger-state array from a remote HTTP endpoint (twice).

    Accepted response formats:
      JSON -> {"fingers": [0, 1, 1, 0, 0]}
      CSV  -> 0,1,1,0,0

    Replace REMOTE_IP / REMOTE_PORT in the configuration section above.
    """

    def __init__(self, endpoint: str, timeout: float) -> None:
        self._endpoint = endpoint
        self._timeout  = timeout

    def fetch(self) -> Optional[List[int]]:
        """Fetch one 5-element array; returns None on any error."""
        try:
            with urllib.request.urlopen(self._endpoint, timeout=self._timeout) as r:
                raw = r.read().decode().strip()
            try:
                fingers = json.loads(raw)["fingers"]
            except (json.JSONDecodeError, KeyError, TypeError):
                fingers = [int(x.strip()) for x in raw.split(",")]
            if len(fingers) == 5 and all(v in (0, 1) for v in fingers):
                return [int(v) for v in fingers]
            log.warning("Unexpected remote payload: %r", raw)
            return None
        except Exception as exc:
            log.error("Remote fetch error: %s", exc)
            return None

    def fetch_password(self) -> Optional[Tuple[List[int], List[int]]]:
        """Two sequential fetches -> (gesture1, gesture2) or None on failure."""
        g1 = self.fetch()
        if g1 is None:
            return None
        g2 = self.fetch()
        if g2 is None:
            return None
        return g1, g2


class ResultSaver:
    """Appends gesture password results (with optional match verdict) to a JSON file."""

    def __init__(self, filepath: str) -> None:
        self._path = filepath

    def save(
        self,
        g1: List[int],
        g2: List[int],
        t1: Optional[List[int]] = None,
        t2: Optional[List[int]] = None,
    ) -> None:
        record: dict = {
            "timestamp": datetime.now().isoformat(),
            "password":  _fmt(g1, g2),
            "gesture1":  g1,
            "gesture2":  g2,
        }
        if t1 is not None and t2 is not None:
            record["target1"] = t1
            record["target2"] = t2
            record["match"]   = (g1 == t1 and g2 == t2)

        records: List[dict] = []
        if os.path.exists(self._path):
            try:
                with open(self._path) as fh:
                    records = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        records.append(record)
        with open(self._path, "w") as fh:
            json.dump(records, fh, indent=2)

        log.info(
            "Result saved  %s  match=%s",
            record["password"],
            record.get("match", "N/A"),
        )


# -- Main orchestration -------------------------------------------------------
def open_camera(camera_source=0):
    """
    Camera setup that works on Windows and Raspberry Pi/Linux.

    Windows:
        Tries DirectShow and MSMF.
    Raspberry Pi/Linux:
        Tries V4L2 and normal OpenCV opening.
    """

    sources_to_try = [
        camera_source,
        0,
        1,
        2,
    ]

    if sys.platform.startswith("win"):
        backends_to_try = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY,
        ]
    else:
        backends_to_try = [
            cv2.CAP_V4L2,
            cv2.CAP_ANY,
        ]

    for source in sources_to_try:
        for backend in backends_to_try:
            log.info("Trying camera source=%s backend=%s", source, backend)

            cap = cv2.VideoCapture(source, backend)

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Try MJPG. This helps many USB cameras and DroidCam-style cameras.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            # Test if frames can actually be read
            for _ in range(20):
                ret, frame = cap.read()
                if ret and frame is not None:
                    log.info("Camera opened successfully: source=%s backend=%s", source, backend)
                    return cap
                time.sleep(0.1)

            log.warning("Camera opened but no frames: source=%s backend=%s", source, backend)
            cap.release()

    raise RuntimeError(
        "Cannot open camera.\n"
        "Try these fixes:\n"
        "1. Close Camera app, Zoom, Teams, DroidCam, or any app using the camera.\n"
        "2. Change CAMERA_SOURCE from 0 to 1 or 2.\n"
        "3. If using DroidCam, make sure DroidCam client is running first.\n"
        "4. Test camera with a simple OpenCV script."
    )
class GesturePasswordServer:
    """
    Two-part gesture password capture and verification.

    State machine:
        FETCH_TARGET -> GESTURE_1 -> GESTURE_2 -> RESULT -> IDLE

    Keys (DISPLAY_WINDOW=True only):
        Q  quit
        R  restart session (re-fetches target from remote)
    """

    def __init__(self) -> None:
        log.info("Initialising GesturePasswordServer...")

        _v    = mp.tasks.vision
        _model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        if not os.path.exists(_model):
            raise RuntimeError(
                "Model file not found: {}\n"
                "Download hand_landmarker.task and place it next to this script.".format(_model)
            )
        options = _v.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_model),
            running_mode=_v.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=MIN_DETECT_CONF,
            min_tracking_confidence=MIN_TRACK_CONF,
        )
        self._landmarker  = _v.HandLandmarker.create_from_options(options)
        self._mp_drawing  = _v.drawing_utils
        self._mp_styles   = _v.drawing_styles
        self._connections = _v.HandLandmarksConnections.HAND_CONNECTIONS

        self._detector       = FingerStateDetector()
        self._debounce_left  = FingerDebounceFilter()
        self._debounce_right = FingerDebounceFilter()
        self._reader         = RemoteGestureReader(REMOTE_ENDPOINT, REMOTE_FETCH_TIMEOUT)
        self._saver          = ResultSaver(RESULTS_FILE)

        open_flag = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
        self._cap = cv2.VideoCapture(CAMERA_SOURCE, open_flag)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera: {!r}".format(CAMERA_SOURCE))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS,          30)

        # Session state
        self._state:        str                 = "FETCH_TARGET"
        self._t1:           Optional[List[int]] = None
        self._t2:           Optional[List[int]] = None
        self._g1:           Optional[List[int]] = None
        self._g2:           Optional[List[int]] = None
        self._match:        Optional[bool]      = None
        self._hold_start:   float               = 0.0
        self._states_left:  List[int]           = [0, 0, 0, 0, 0]
        self._states_right: List[int]           = [0, 0, 0, 0, 0]

        # Metrics
        self._fps:        float = 0.0
        self._t_fps:      float = time.perf_counter()
        self._start_time: float = time.perf_counter()
        self._frames:     int   = 0

        log.info("Ready. Remote endpoint: %s", REMOTE_ENDPOINT)

    # -- Public ---------------------------------------------------------------

    def run(self) -> None:
        log.info("Running. Press Q to quit, R to restart.")
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    log.warning("Frame capture failed -- camera disconnected?")
                    break
                frame = cv2.flip(frame, 1)  # mirror so left/right match the user's view
                self._process(frame)
                if DISPLAY_WINDOW:
                    cv2.imshow("Gesture Password", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("r"):
                        self._reset()
        finally:
            self._shutdown()

    # -- Per-frame pipeline ---------------------------------------------------

    def _process(self, frame) -> None:
        self._frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        both_detected = False
        if len(result.hand_landmarks) == 2:
            # Sort the two hands left-to-right by wrist x-coordinate
            hands = sorted(result.hand_landmarks, key=lambda lm: lm[0].x)
            # Left hand: reverse so the array reads left-to-right on screen (pinky→thumb → thumb→pinky)
            self._states_left  = self._debounce_left.push(self._detector.detect(hands[0])[::-1])
            self._states_right = self._debounce_right.push(self._detector.detect(hands[1]))
            both_detected = True

        self._step(both_detected)
        if DISPLAY_WINDOW:
            self._draw(frame, result, both_detected)
        self._tick_fps()

    # -- State machine --------------------------------------------------------

    def _step(self, both_detected: bool) -> None:
        now = time.perf_counter()

        if self._state == "FETCH_TARGET":
            fetched = self._reader.fetch_password()
            if fetched is not None:
                self._t1, self._t2 = fetched
                log.info("Target password (remote): %s", _fmt(self._t1, self._t2))
            else:
                self._t1, self._t2 = list(TEST_TARGET_1), list(TEST_TARGET_2)
                log.warning("Remote unavailable -- using test target: %s", _fmt(self._t1, self._t2))
            self._state = "CAPTURE"

        elif self._state == "CAPTURE":
            if both_detected:
                if self._hold_start == 0.0:
                    self._hold_start = now
                elif now - self._hold_start >= GESTURE_HOLD_SECONDS:
                    self._g1 = list(self._states_left)
                    self._g2 = list(self._states_right)
                    log.info("Captured  Left: %s  Right: %s", self._g1, self._g2)
                    self._hold_start = 0.0
                    self._debounce_left.reset()
                    self._debounce_right.reset()
                    self._state = "RESULT"
            else:
                self._hold_start = 0.0

        elif self._state == "RESULT":
            if self._g1 is not None and self._g2 is not None:
                self._saver.save(self._g1, self._g2, self._t1, self._t2)
                matched = (
                    (self._g1 == self._t1 and self._g2 == self._t2)
                    if (self._t1 is not None and self._t2 is not None) else None
                )
                if matched is not False:
                    self._match = matched
                    log.info("Password: %s  RIGHT", _fmt(self._g1, self._g2))
                    self._state = "IDLE"
                else:
                    log.info("Password: %s  no match -- retrying", _fmt(self._g1, self._g2))
                    self._g1 = None
                    self._g2 = None
                    self._debounce_left.reset()
                    self._debounce_right.reset()
                    self._state = "CAPTURE"

    # -- Overlay --------------------------------------------------------------

    def _draw(self, frame, result, both_detected: bool) -> None:
        h, w = frame.shape[:2]
        now  = time.perf_counter()

        # Hand skeletons
        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame, hand_lm, self._connections,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )

        # Top HUD (dark bar)
        cv2.rectangle(frame, (0, 0), (w, 110), (15, 15, 15), -1)

        color = {
            "FETCH_TARGET": (200, 200,   0),
            "CAPTURE":      (  0, 200, 255),
            "RESULT":       (  0, 255,   0) if self._match is not False else (0, 0, 255),
            "IDLE":         (150, 150, 150),
        }.get(self._state, (255, 255, 255))

        prompt = {
            "FETCH_TARGET": "Fetching target from server...",
            "CAPTURE":      "Show BOTH hands and hold steady",
            "RESULT":       "MATCH!" if self._match else (
                                "MISMATCH" if self._match is False else "Password saved"
                            ),
            "IDLE":         "Done -- press R to restart, Q to quit",
        }.get(self._state, "")

        cv2.putText(frame, prompt, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        # Live finger states for both hands
        if self._state == "CAPTURE":
            left_color  = (0, 255, 80) if both_detected else (80, 80, 200)
            right_color = (0, 255, 80) if both_detected else (80, 80, 200)

            cv2.putText(
                frame,
                "L:[{}]".format(" ".join(map(str, self._states_left))),
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, left_color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "R:[{}]".format(" ".join(map(str, self._states_right))),
                (w // 2, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, right_color, 2, cv2.LINE_AA,
            )

            # Hold progress bar (only fills when both hands are present)
            if both_detected and self._hold_start:
                pct   = min(1.0, (now - self._hold_start) / GESTURE_HOLD_SECONDS)
                bar_w = max(1, int(pct * (w - 20)))
                cv2.rectangle(frame, (10, 90), (w - 10, 100), (60, 60, 60), -1)
                cv2.rectangle(frame, (10, 90), (10 + bar_w, 100), (0, 220, 80), -1)
            elif not both_detected:
                missing = "Need {} more hand(s)".format(2 - len(result.hand_landmarks))
                cv2.putText(frame, missing, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 200), 1, cv2.LINE_AA)

        # Bottom bar
        cv2.rectangle(frame, (0, h - 55), (w, h), (15, 15, 15), -1)

        if self._t1 is not None and self._t2 is not None:
            cv2.putText(
                frame, "Target: " + _fmt(self._t1, self._t2),
                (10, h - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 255), 1, cv2.LINE_AA,
            )

        if self._g1 is not None and self._g2 is not None:
            cv2.putText(
                frame, "Pass:   " + _fmt(self._g1, self._g2),
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA,
            )

        # Big centered verdict
        if self._state in ("RESULT", "IDLE") and self._match is not None:
            verdict      = "RIGHT" if self._match else "WRONG"
            verdict_color = (0, 230, 0) if self._match else (0, 0, 230)
            font_scale   = 4.0
            thickness    = 8
            (tw, th), _  = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            tx = (w - tw) // 2
            ty = (h + th) // 2
            # Dark shadow for readability
            cv2.putText(frame, verdict, (tx + 3, ty + 3),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (10, 10, 10), thickness + 4, cv2.LINE_AA)
            cv2.putText(frame, verdict, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, verdict_color, thickness, cv2.LINE_AA)

        cv2.putText(
            frame, "FPS {:.1f}".format(self._fps),
            (w - 95, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA,
        )

    # -- Utilities ------------------------------------------------------------

    def _reset(self) -> None:
        self._state      = "FETCH_TARGET"
        self._t1         = None
        self._t2         = None
        self._g1         = None
        self._g2         = None
        self._match      = None
        self._hold_start   = 0.0
        self._states_left  = [0, 0, 0, 0, 0]
        self._states_right = [0, 0, 0, 0, 0]
        self._debounce_left.reset()
        self._debounce_right.reset()
        log.info("Session reset.")

    def _tick_fps(self) -> None:
        now = time.perf_counter()
        dt  = now - self._t_fps
        self._t_fps = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 / dt

    def _shutdown(self) -> None:
        log.info("Shutdown after %d frames.", self._frames)
        self._landmarker.close()
        self._cap.release()
        if DISPLAY_WINDOW:
            cv2.destroyAllWindows()


# -- Entry point --------------------------------------------------------------

def main() -> int:
    try:
        GesturePasswordServer().run()
    except KeyboardInterrupt:
        log.info("Interrupted.")
    except RuntimeError as exc:
        log.critical("Fatal: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())