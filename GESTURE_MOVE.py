from pathlib import Path
import time

import cv2
import mediapipe as mp
from djitellopy import Tello


MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

MINIMUM_BATTERY = 50
MOVEMENT_DISTANCE = 20
DEAD_ZONE = 0.30

CENTRE_HOLD_SECONDS = 0.7
MOVEMENT_HOLD_SECONDS = 1.0
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


def hand_is_open(landmarks):
    finger_joints = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    ]

    extended_fingers = 0

    for fingertip, middle_joint in finger_joints:
        if landmarks[fingertip].y < landmarks[middle_joint].y:
            extended_fingers += 1

    return extended_fingers >= 3


def calculate_horizontal_direction(center_x, width):
    horizontal = (center_x - width / 2) / (width / 2)

    if abs(horizontal) < DEAD_ZONE:
        return "HOVER"

    if horizontal < 0:
        return "LEFT"

    return "RIGHT"


def land_drone(tello, reason):
    print(f"Landing: {reason}")

    try:
        tello.land()
        return True
    except Exception as error:
        print(f"Landing failed: {error}")
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

try:
    print("Connecting to Tello...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    if battery < MINIMUM_BATTERY:
        print(f"Battery must be at least {MINIMUM_BATTERY}%.")
        raise SystemExit

    confirmation = input(
        "Type GESTURE_MOVE to begin, or anything else to cancel: "
    )

    if confirmation != "GESTURE_MOVE":
        print("Test cancelled.")
        raise SystemExit

    tello.streamon()
    stream_started = True
    frame_reader = tello.get_frame_read()

    print("Taking off...")
    flight_may_be_active = True
    tello.takeoff()

    flight_started = time.monotonic()
    last_hand_seen = flight_started

    movement_ready = False
    centre_started = None
    candidate_direction = None
    direction_started = None

    while flight_may_be_active:
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

        now = time.monotonic()
        open_palm_active = False
        direction = "HOVER"
        progress = 0.0

        if result.hand_landmarks:
            last_hand_seen = now
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
                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 255, 0),
                    -1,
                )

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

            model_open_palm = False

            if result.gestures and result.gestures[0]:
                gesture = result.gestures[0][0]

                model_open_palm = (
                    gesture.category_name == "Open_Palm"
                    and gesture.score >= 0.45
                )

            open_palm_active = (
                model_open_palm
                or hand_is_open(landmarks)
            )

            if open_palm_active:
                direction = calculate_horizontal_direction(
                    center_x,
                    width,
                )

        if open_palm_active and direction == "HOVER":
            candidate_direction = None
            direction_started = None

            if centre_started is None:
                centre_started = now

            centre_progress = (
                now - centre_started
            ) / CENTRE_HOLD_SECONDS

            progress = min(centre_progress, 1.0)

            if centre_progress >= 1.0:
                movement_ready = True

        elif open_palm_active and movement_ready:
            centre_started = None

            if direction != candidate_direction:
                candidate_direction = direction
                direction_started = now

            held_for = now - direction_started
            progress = min(
                held_for / MOVEMENT_HOLD_SECONDS,
                1.0,
            )

            if held_for >= MOVEMENT_HOLD_SECONDS:
                print(f"Confirmed movement: {direction}")

                if direction == "LEFT":
                    tello.move_left(MOVEMENT_DISTANCE)

                elif direction == "RIGHT":
                    tello.move_right(MOVEMENT_DISTANCE)

                time.sleep(2)

                landed = land_drone(
                    tello,
                    "movement test complete",
                )

                if landed:
                    flight_may_be_active = False

                break

        else:
            centre_started = None
            candidate_direction = None
            direction_started = None
            progress = 0.0

        if (
            now - last_hand_seen
            >= NO_HAND_LANDING_SECONDS
        ):
            landed = land_drone(tello, "hand lost")

            if landed:
                flight_may_be_active = False

            break

        if (
            now - flight_started
            >= MAXIMUM_FLIGHT_SECONDS
        ):
            landed = land_drone(
                tello,
                "maximum flight time reached",
            )

            if landed:
                flight_may_be_active = False

            break

        left_boundary = int(
            width / 2 - width / 2 * DEAD_ZONE
        )

        right_boundary = int(
            width / 2 + width / 2 * DEAD_ZONE
        )

        cv2.rectangle(
            frame,
            (left_boundary, 170),
            (right_boundary, 330),
            (0, 255, 255),
            3,
        )

        status = (
            "READY: MOVE PALM LEFT OR RIGHT"
            if movement_ready
            else "CENTRE OPEN PALM TO ENABLE"
        )

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"DIRECTION: {direction}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        bar_width = 250
        filled_width = int(bar_width * progress)
        percentage = int(progress * 100)

        cv2.rectangle(
            frame,
            (20, 95),
            (20 + bar_width, 118),
            (100, 100, 100),
            2,
        )

        cv2.rectangle(
            frame,
            (20, 95),
            (20 + filled_width, 118),
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            frame,
            f"{percentage}%",
            (280, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "L = land | Q = quit and land",
            (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

        cv2.imshow(
            "PalmPilot Gesture Movement Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("l"), ord("L")):
            landed = land_drone(
                tello,
                "keyboard command",
            )

            if landed:
                flight_may_be_active = False

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
    print("Test closed.")
    