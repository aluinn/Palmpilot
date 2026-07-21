from pathlib import Path
import time

import cv2
import mediapipe as mp
from djitellopy import Tello


MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
MIN_CONFIDENCE = 0.65

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def get_transition(state, gesture):
    """Return action, next state and required hold time."""

    if state == "DISARMED" and gesture == "Victory":
        return "ARM CONTROLS", "ARMED", 2.0

    if state == "ARMED" and gesture == "Closed_Fist":
        return "DISARM CONTROLS", "DISARMED", 2.0

    if state == "ARMED" and gesture == "Thumb_Up":
        return "SIMULATED TAKEOFF", "FLYING", 2.0

    if state == "FLYING" and gesture == "Thumb_Down":
        return "SIMULATED LANDING", "ARMED", 1.5

    return None


def get_instructions(state):
    if state == "DISARMED":
        return "Hold VICTORY for 2 seconds to arm"

    if state == "ARMED":
        return "THUMB UP: takeoff preview | FIST: disarm"

    return "THUMB DOWN: landing preview | OPEN PALM: hover"


options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
)

recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

tello = Tello()
stream_started = False

state = "DISARMED"
candidate_gesture = None
candidate_started = time.monotonic()
transition_triggered = False
last_action = "No action yet"

try:
    print("Connecting to Tello...")
    tello.connect()
    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    tello.streamon()
    stream_started = True
    frame_reader = tello.get_frame_read()

    print("Preview controller running — no flight commands are enabled")
    print("Press Q to close")

    while True:
        frame = frame_reader.frame

        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = recognizer.recognize(mp_image)

        gesture_name = "No hand"
        confidence = 0.0

        if result.hand_landmarks:
            height, width, _ = frame.shape
            landmarks = result.hand_landmarks[0]
            points = []

            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 150, 0),
                    2,
                )

            for point in points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

        if result.gestures and result.gestures[0]:
            best_gesture = result.gestures[0][0]
            gesture_name = best_gesture.category_name or "Unknown"
            confidence = best_gesture.score

        stable_gesture = (
            gesture_name
            if confidence >= MIN_CONFIDENCE
            else "Unclear"
        )

        now = time.monotonic()

        if stable_gesture != candidate_gesture:
            candidate_gesture = stable_gesture
            candidate_started = now
            transition_triggered = False

        held_for = now - candidate_started
        transition = get_transition(state, stable_gesture)
        progress = 0.0

        if transition is not None:
            action, next_state, required_time = transition
            progress = min(held_for / required_time, 1.0)

            if held_for >= required_time and not transition_triggered:
                state = next_state
                last_action = f"PREVIEW: {action}"
                transition_triggered = True

        if state == "FLYING" and stable_gesture == "Open_Palm":
            last_action = "PREVIEW: HOVER"

        state_colour = {
            "DISARMED": (0, 0, 255),
            "ARMED": (0, 165, 255),
            "FLYING": (0, 255, 0),
        }[state]

        cv2.putText(
            frame,
            f"STATE: {state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            state_colour,
            2,
        )

        cv2.putText(
            frame,
            f"GESTURE: {gesture_name} ({confidence:.0%})",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            last_action,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            get_instructions(state),
            (20, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        bar_width = 300
        filled_width = int(bar_width * progress)

        cv2.rectangle(frame, (20, 135), (320, 160), (100, 100, 100), 2)
        cv2.rectangle(
            frame,
            (20, 135),
            (20 + filled_width, 160),
            (0, 255, 255),
            -1,
        )

        cv2.imshow("PalmPilot Safety Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if stream_started:
        tello.streamoff()

    tello.end()
    recognizer.close()
    cv2.destroyAllWindows()
