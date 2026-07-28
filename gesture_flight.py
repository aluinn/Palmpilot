from pathlib import Path
import time

import cv2
import mediapipe as mp
from djitellopy import Tello


MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

MINIMUM_BATTERY = 50
MINIMUM_CONFIDENCE = 0.70
MAXIMUM_FLIGHT_SECONDS = 15
NO_HAND_LANDING_SECONDS = 5

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def get_transition(state, gesture):
    if state == "DISARMED" and gesture == "Victory":
        return "ARM", 2.0

    if state == "ARMED" and gesture == "Closed_Fist":
        return "DISARM", 2.0

    if state == "ARMED" and gesture == "Thumb_Up":
        return "TAKEOFF", 2.0

    if state == "FLYING" and gesture == "Thumb_Down":
        return "LAND", 1.5

    return None


def get_instructions(state):
    if state == "DISARMED":
        return "Hold VICTORY to arm"

    if state == "ARMED":
        return "Hold THUMB UP to take off | FIST to disarm"

    return "Hold THUMB DOWN to land | Press L to land"


def land_drone(tello, reason):
    print(f"Landing: {reason}")

    try:
        tello.land()
        return True
    except Exception as error:
        print(f"Landing command failed: {error}")
        return False


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
flight_may_be_active = False

state = "DISARMED"
candidate_gesture = None
candidate_started = time.monotonic()
transition_triggered = False

flight_started = None
last_hand_seen = time.monotonic()
last_action = "Waiting for arming gesture"

try:
    print("Connecting to Tello...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    if battery < MINIMUM_BATTERY:
        print(f"Battery must be at least {MINIMUM_BATTERY}%.")
        raise SystemExit

    tello.streamon()
    stream_started = True
    frame_reader = tello.get_frame_read()

    print("Gesture flight controller ready")
    print("L = land immediately")
    print("Q = quit and land")

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

        now = time.monotonic()
        gesture_name = "No hand"
        confidence = 0.0

        if result.hand_landmarks:
            last_hand_seen = now

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

        if confidence >= MINIMUM_CONFIDENCE:
            stable_gesture = gesture_name
        else:
            stable_gesture = "Unclear"

        if stable_gesture != candidate_gesture:
            candidate_gesture = stable_gesture
            candidate_started = now
            transition_triggered = False

        held_for = now - candidate_started
        transition = get_transition(state, stable_gesture)
        progress = 0.0

        if transition is not None:
            action, required_time = transition
            progress = min(held_for / required_time, 1.0)

            if held_for >= required_time and not transition_triggered:
                transition_triggered = True

                if action == "ARM":
                    state = "ARMED"
                    last_action = "CONTROLS ARMED"

                elif action == "DISARM":
                    state = "DISARMED"
                    last_action = "CONTROLS DISARMED"

                elif action == "TAKEOFF":
                    last_action = "TAKING OFF"
                    flight_may_be_active = True

                    try:
                        tello.takeoff()
                    except Exception as error:
                        print(f"Takeoff failed: {error}")
                        landed = land_drone(
                            tello,
                            "takeoff command error",
                        )
                        flight_may_be_active = not landed
                        state = "DISARMED"
                        last_action = "TAKEOFF FAILED"

                        if not landed:
                            break
                    else:
                        state = "FLYING"
                        flight_started = time.monotonic()
                        last_hand_seen = flight_started
                        last_action = "HOVERING"

                elif action == "LAND":
                    landed = land_drone(
                        tello,
                        "thumbs-down gesture",
                    )

                    if landed:
                        flight_may_be_active = False
                        state = "DISARMED"
                        flight_started = None
                        last_action = "LANDED AND DISARMED"
                    else:
                        break

        # Automatic safety landing if the hand disappears.
        if (
            state == "FLYING"
            and now - last_hand_seen >= NO_HAND_LANDING_SECONDS
        ):
            landed = land_drone(tello, "hand lost")

            if landed:
                flight_may_be_active = False
                state = "DISARMED"
                flight_started = None
                last_action = "AUTO-LANDED: HAND LOST"
                transition_triggered = True
            else:
                break

        # Automatic safety landing after the maximum flight time.
        if (
            state == "FLYING"
            and flight_started is not None
            and now - flight_started >= MAXIMUM_FLIGHT_SECONDS
        ):
            landed = land_drone(tello, "maximum flight time reached")

            if landed:
                flight_may_be_active = False
                state = "DISARMED"
                flight_started = None
                last_action = "AUTO-LANDED: TIME LIMIT"
                transition_triggered = True
            else:
                break

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
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            get_instructions(state),
            (20, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            "SAFETY: L = LAND | Q = QUIT AND LAND",
            (20, 465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
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

        cv2.imshow("PalmPilot Gesture Flight", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("l"), ord("L")) and flight_may_be_active:
            landed = land_drone(tello, "keyboard command")

            if landed:
                flight_may_be_active = False
                state = "DISARMED"
                flight_started = None
                last_action = "KEYBOARD LANDING COMPLETE"
            else:
                break

        if key in (ord("q"), ord("Q")):
            break

finally:
    if flight_may_be_active:
        land_drone(tello, "program closing")

    if stream_started:
        try:
            tello.streamoff()
        except Exception as error:
            print(f"Could not stop video stream: {error}")

    tello.end()
    recognizer.close()
    cv2.destroyAllWindows()
    print("Controller closed.")