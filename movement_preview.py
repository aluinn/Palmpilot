from pathlib import Path
import time

import cv2
import mediapipe as mp
from djitellopy import Tello


MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

DIRECTION_HOLD_SECONDS = 0.6
DEAD_ZONE = 0.30
MODEL_OPEN_PALM_CONFIDENCE = 0.45

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def calculate_direction(center_x, center_y, width, height):
    """Choose one direction based on the palm's screen position."""

    horizontal = (center_x - width / 2) / (width / 2)
    vertical = (center_y - height / 2) / (height / 2)

    if abs(horizontal) < DEAD_ZONE and abs(vertical) < DEAD_ZONE:
        return "HOVER"

    # Only choose one axis at a time.
    if abs(horizontal) >= abs(vertical):
        if horizontal < 0:
            return "LEFT"

        return "RIGHT"

    if vertical < 0:
        return "UP"

    return "DOWN"


def hand_is_open(landmarks):
    """Detect an upright open hand using finger landmarks."""

    finger_joints = [
        (8, 6),    # Index finger
        (12, 10),  # Middle finger
        (16, 14),  # Ring finger
        (20, 18),  # Little finger
    ]

    extended_fingers = 0

    for fingertip, middle_joint in finger_joints:
        if landmarks[fingertip].y < landmarks[middle_joint].y:
            extended_fingers += 1

    return extended_fingers >= 3


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

candidate_direction = "HOVER"
candidate_started = time.monotonic()
confirmed_direction = "HOVER"

try:
    print("Connecting to Tello...")
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    stream_started = True
    frame_reader = tello.get_frame_read()

    print("Movement preview running")
    print("No flight or movement commands are enabled")
    print("Press Q to close")

    while True:
        frame = frame_reader.frame

        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = recognizer.recognize(mp_image)

        gesture_name = "No hand"
        gesture_confidence = 0.0
        raw_direction = "HOVER"
        open_palm_is_active = False

        # Read MediaPipe's built-in gesture result.
        if result.gestures and result.gestures[0]:
            best_gesture = result.gestures[0][0]
            gesture_name = best_gesture.category_name or "Unknown"
            gesture_confidence = best_gesture.score

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            points = []

            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

            # Draw the hand skeleton.
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 150, 0),
                    2,
                )

            for point in points:
                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 255, 0),
                    -1,
                )

            # Use landmarks around the palm to calculate its centre.
            palm_indices = [0, 5, 9, 13, 17]

            center_x = int(
                sum(points[index][0] for index in palm_indices)
                / len(palm_indices)
            )

            center_y = int(
                sum(points[index][1] for index in palm_indices)
                / len(palm_indices)
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                12,
                (0, 0, 255),
                -1,
            )

            # Accept either MediaPipe's label or our landmark check.
            model_detected_open_palm = (
                gesture_name == "Open_Palm"
                and gesture_confidence
                >= MODEL_OPEN_PALM_CONFIDENCE
            )

            landmark_detected_open_palm = hand_is_open(landmarks)

            open_palm_is_active = (
                model_detected_open_palm
                or landmark_detected_open_palm
            )

            if open_palm_is_active:
                raw_direction = calculate_direction(
                    center_x,
                    center_y,
                    width,
                    height,
                )

        now = time.monotonic()

        if open_palm_is_active:
            if raw_direction != candidate_direction:
                candidate_direction = raw_direction
                candidate_started = now

            held_for = now - candidate_started

            progress = min(
                held_for / DIRECTION_HOLD_SECONDS,
                1.0,
            )

            if held_for >= DIRECTION_HOLD_SECONDS:
                confirmed_direction = candidate_direction

        else:
            # Without an open palm, safely reset to hover.
            raw_direction = "HOVER"
            candidate_direction = "HOVER"
            confirmed_direction = "HOVER"
            candidate_started = now
            progress = 0.0

        # Calculate the central hover-zone boundaries.
        left_boundary = int(
            width / 2 - width / 2 * DEAD_ZONE
        )

        right_boundary = int(
            width / 2 + width / 2 * DEAD_ZONE
        )

        top_boundary = int(
            height / 2 - height / 2 * DEAD_ZONE
        )

        bottom_boundary = int(
            height / 2 + height / 2 * DEAD_ZONE
        )

        # Draw a complete yellow outline with no shading.
        cv2.rectangle(
            frame,
            (left_boundary, top_boundary),
            (right_boundary, bottom_boundary),
            (0, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            "HOVER ZONE",
            (left_boundary + 25, top_boundary + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        direction_colours = {
            "HOVER": (0, 255, 0),
            "LEFT": (255, 0, 0),
            "RIGHT": (255, 0, 0),
            "UP": (255, 0, 255),
            "DOWN": (255, 0, 255),
        }

        palm_status = (
            "OPEN PALM ACTIVE"
            if open_palm_is_active
            else "SHOW AN OPEN PALM"
        )

        cv2.putText(
            frame,
            "MOVEMENT PREVIEW ONLY",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            palm_status,
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                (0, 255, 0)
                if open_palm_is_active
                else (0, 0, 255)
            ),
            2,
        )

        cv2.putText(
            frame,
            f"RAW: {raw_direction}",
            (20, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"CONFIRMED: {confirmed_direction}",
            (20, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            direction_colours[confirmed_direction],
            2,
        )

        # Confirmation bar that fills from 0% to 100%.
        bar_width = 250
        filled_width = int(bar_width * progress)
        percentage = int(progress * 100)

        cv2.rectangle(
            frame,
            (20, 140),
            (20 + bar_width, 162),
            (100, 100, 100),
            2,
        )

        cv2.rectangle(
            frame,
            (20, 140),
            (20 + filled_width, 162),
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            frame,
            f"{percentage}%",
            (280, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Keep fingers upright and fully visible",
            (20, 438),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            "Q = quit",
            (20, 462),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow(
            "PalmPilot Movement Preview",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

finally:
    if stream_started:
        try:
            tello.streamoff()
        except Exception as error:
            print(f"Could not stop video stream: {error}")

    tello.end()
    recognizer.close()
    cv2.destroyAllWindows()
    print("Movement preview closed.")

    

