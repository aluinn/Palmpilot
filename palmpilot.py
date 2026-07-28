from pathlib import Path
import time

import cv2
import mediapipe as mp
from djitellopy import Tello


# -------------------- SETTINGS --------------------

MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

LAND_AT_BATTERY = 20
NO_HAND_LAND_SECONDS = 30

MAX_LEFT_RIGHT_SPEED = 24
MAX_UP_DOWN_SPEED = 20
MIN_MOVEMENT_SPEED = 10

DEPTH_SPEED = 14
DEPTH_MAX_SECONDS = 2.0
DEPTH_GESTURE_HOLD_SECONDS = 0.25
DEPTH_RESET_HOLD_SECONDS = 0.5

DEAD_ZONE_X = 0.09
DEAD_ZONE_Y = 0.10

SMOOTHING = 0.28
RC_INTERVAL = 0.05
ACTIVATION_HOLD_SECONDS = 1.0
GESTURE_CONFIDENCE = 0.35


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def speed_from_error(error, dead_zone, maximum_speed):
    """Convert palm distance from the centre into speed."""

    if abs(error) <= dead_zone:
        return 0

    usable_area = 0.5 - dead_zone

    strength = min(
        1.0,
        (abs(error) - dead_zone) / usable_area,
    )

    speed = MIN_MOVEMENT_SPEED + (
        maximum_speed - MIN_MOVEMENT_SPEED
    ) * strength

    if error < 0:
        speed = -speed

    return int(round(speed))


def hand_is_open(landmarks, gesture_name, gesture_score):
    """Detect an open palm using the model and finger positions."""

    model_says_open = (
        gesture_name == "Open_Palm"
        and gesture_score >= 0.40
    )

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    ]

    raised_fingers = 0

    for fingertip, middle_joint in finger_pairs:
        if landmarks[fingertip].y < landmarks[middle_joint].y - 0.01:
            raised_fingers += 1

    geometry_says_open = raised_fingers >= 3

    return model_says_open or geometry_says_open


def forward_gesture_detected(
    landmarks,
    gesture_name,
    gesture_score,
):
    """Detect Victory or Pointing Up as the forward command."""

    model_detected_forward = (
        gesture_score >= GESTURE_CONFIDENCE
        and gesture_name in ("Victory", "Pointing_Up")
    )

    index_up = landmarks[8].y < landmarks[6].y - 0.01
    middle_up = landmarks[12].y < landmarks[10].y - 0.01

    middle_folded = landmarks[12].y > landmarks[10].y
    ring_folded = landmarks[16].y > landmarks[14].y
    little_folded = landmarks[20].y > landmarks[18].y

    victory_shape = (
        index_up
        and middle_up
        and ring_folded
        and little_folded
    )

    pointing_shape = (
        index_up
        and middle_folded
        and ring_folded
        and little_folded
    )

    return (
        model_detected_forward
        or victory_shape
        or pointing_shape
    )


def draw_hand(frame, landmarks):
    """Draw hand landmarks and connections."""

    height, width = frame.shape[:2]

    for start, end in HAND_CONNECTIONS:
        start_point = (
            int(landmarks[start].x * width),
            int(landmarks[start].y * height),
        )

        end_point = (
            int(landmarks[end].x * width),
            int(landmarks[end].y * height),
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    for landmark in landmarks:
        point = (
            int(landmark.x * width),
            int(landmark.y * height),
        )

        cv2.circle(
            frame,
            point,
            4,
            (0, 255, 0),
            -1,
        )


def main():
    if not MODEL_PATH.exists():
        print("gesture_recognizer.task was not found.")
        print("Put it in the same folder as palmpilot.py.")
        return

    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    recognizer = (
        mp.tasks.vision.GestureRecognizer.create_from_options(
            options
        )
    )

    tello = Tello()

    flying = False
    streaming = False
    landing_reason = None

    tracking_active = False
    activation_started = None

    smoothed_x = None
    smoothed_y = None
    last_open_palm_time = 0

    last_hand_seen = None
    last_rc_sent = 0
    last_battery_check = 0
    previous_timestamp = -1

    battery = 0

    # Forward/backward gesture state.
    depth_candidate = None
    depth_candidate_started = None
    depth_motion_started = None
    depth_locked = False
    depth_reset_started = None

    try:
        print("Connecting to Tello...")
        tello.connect()

        battery = tello.get_battery()
        print(f"Battery: {battery}%")

        if battery <= LAND_AT_BATTERY:
            print(
                f"Warning: battery is already at or below "
                f"{LAND_AT_BATTERY}%."
            )
            print("The drone will land shortly after takeoff.")

        print()
        print("Controls:")
        print("Open palm = left/right/up/down")
        print("Victory sign = forward")
        print("Pointing Up = forward")
        print("Closed fist = backward")
        print("L or Q = land")
        print()

        confirmation = input(
            "Clear the area, attach propeller guards, "
            "then type TAKEOFF: "
        )

        if confirmation.strip().upper() != "TAKEOFF":
            print("Takeoff cancelled.")
            return

        print("Starting camera...")
        tello.streamon()
        streaming = True

        frame_reader = tello.get_frame_read()
        time.sleep(2)

        print("Taking off...")
        tello.takeoff()
        flying = True

        last_hand_seen = time.monotonic()
        time.sleep(2)

        print("Hold an open palm inside the yellow box.")

        while flying:
            now = time.monotonic()
            raw_frame = frame_reader.frame

            if raw_frame is None or raw_frame.size == 0:
                continue

            rgb_frame = cv2.resize(
                raw_frame,
                (960, 720),
            )

            # Mirror the camera display.
            rgb_frame = cv2.flip(rgb_frame, 1)

            timestamp = max(
                previous_timestamp + 1,
                int(time.monotonic() * 1000),
            )
            previous_timestamp = timestamp

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = recognizer.recognize_for_video(
                mp_image,
                timestamp,
            )

            display = cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGB2BGR,
            )

            height, width = display.shape[:2]

            zone_left = int((0.5 - DEAD_ZONE_X) * width)
            zone_right = int((0.5 + DEAD_ZONE_X) * width)
            zone_top = int((0.5 - DEAD_ZONE_Y) * height)
            zone_bottom = int((0.5 + DEAD_ZONE_Y) * height)

            cv2.rectangle(
                display,
                (zone_left, zone_top),
                (zone_right, zone_bottom),
                (0, 255, 255),
                3,
            )

            left_right_speed = 0
            forward_backward_speed = 0
            up_down_speed = 0

            status = "HOVER"
            gesture_text = "No hand"

            activation_progress = 0.0
            depth_progress = 0.0
            reset_progress = 0.0

            hand_detected = bool(result.hand_landmarks)

            if hand_detected:
                last_hand_seen = now

                landmarks = result.hand_landmarks[0]
                draw_hand(display, landmarks)

                gesture_name = "Unknown"
                gesture_score = 0.0

                if result.gestures and result.gestures[0]:
                    gesture = result.gestures[0][0]
                    gesture_name = gesture.category_name
                    gesture_score = gesture.score

                gesture_text = (
                    f"{gesture_name}: "
                    f"{gesture_score * 100:.0f}%"
                )

                open_palm = hand_is_open(
                    landmarks,
                    gesture_name,
                    gesture_score,
                )

                recognised_depth_gesture = None

                if forward_gesture_detected(
                    landmarks,
                    gesture_name,
                    gesture_score,
                ):
                    recognised_depth_gesture = "FORWARD"

                elif (
                    gesture_score >= GESTURE_CONFIDENCE
                    and gesture_name == "Closed_Fist"
                ):
                    recognised_depth_gesture = "BACKWARD"

                palm_x = landmarks[9].x
                palm_y = landmarks[9].y

                palm_point = (
                    int(palm_x * width),
                    int(palm_y * height),
                )

                cv2.circle(
                    display,
                    palm_point,
                    10,
                    (0, 0, 255),
                    -1,
                )

                if open_palm:
                    if (
                        smoothed_x is None
                        or now - last_open_palm_time > 0.5
                    ):
                        smoothed_x = palm_x
                        smoothed_y = palm_y
                    else:
                        smoothed_x = (
                            SMOOTHING * palm_x
                            + (1 - SMOOTHING) * smoothed_x
                        )

                        smoothed_y = (
                            SMOOTHING * palm_y
                            + (1 - SMOOTHING) * smoothed_y
                        )

                    last_open_palm_time = now

                # Activate tracking with an open palm in the centre.
                if not tracking_active:
                    status = "HOLD OPEN PALM IN YELLOW BOX"

                    if open_palm:
                        palm_is_centred = (
                            abs(smoothed_x - 0.5) <= DEAD_ZONE_X
                            and abs(smoothed_y - 0.5)
                            <= DEAD_ZONE_Y
                        )

                        if palm_is_centred:
                            if activation_started is None:
                                activation_started = now

                            activation_progress = min(
                                1.0,
                                (now - activation_started)
                                / ACTIVATION_HOLD_SECONDS,
                            )

                            if activation_progress >= 1.0:
                                tracking_active = True
                                activation_started = None
                                status = "TRACKING ACTIVE"
                        else:
                            activation_started = None
                    else:
                        activation_started = None

                # Require an open-palm reset after depth movement.
                elif depth_locked:
                    status = "SHOW OPEN PALM TO RESET DEPTH"

                    if open_palm:
                        if depth_reset_started is None:
                            depth_reset_started = now

                        reset_progress = min(
                            1.0,
                            (now - depth_reset_started)
                            / DEPTH_RESET_HOLD_SECONDS,
                        )

                        if reset_progress >= 1.0:
                            depth_locked = False
                            depth_reset_started = None
                            depth_candidate = None
                            depth_candidate_started = None
                            depth_motion_started = None
                            status = "DEPTH RESET - HOVER"
                    else:
                        depth_reset_started = None

                # Victory or Pointing Up moves forward.
                # Closed fist moves backward.
                elif recognised_depth_gesture is not None:
                    if depth_candidate != recognised_depth_gesture:
                        depth_candidate = recognised_depth_gesture
                        depth_candidate_started = now
                        depth_motion_started = None

                    gesture_hold_time = (
                        now - depth_candidate_started
                    )

                    if (
                        gesture_hold_time
                        < DEPTH_GESTURE_HOLD_SECONDS
                    ):
                        status = (
                            f"HOLD {recognised_depth_gesture} "
                            f"GESTURE"
                        )

                    else:
                        if depth_motion_started is None:
                            depth_motion_started = now

                        movement_time = (
                            now - depth_motion_started
                        )

                        depth_progress = min(
                            1.0,
                            movement_time / DEPTH_MAX_SECONDS,
                        )

                        if movement_time >= DEPTH_MAX_SECONDS:
                            depth_locked = True
                            forward_backward_speed = 0

                            status = (
                                "DEPTH LIMIT - "
                                "SHOW OPEN PALM"
                            )

                        elif recognised_depth_gesture == "FORWARD":
                            forward_backward_speed = DEPTH_SPEED
                            status = "MOVING FORWARD"

                        elif recognised_depth_gesture == "BACKWARD":
                            forward_backward_speed = -DEPTH_SPEED
                            status = "MOVING BACKWARD"

                # Open palm controls left/right/up/down.
                elif open_palm:
                    depth_candidate = None
                    depth_candidate_started = None
                    depth_motion_started = None

                    # Reversed so your hand and drone move
                    # in the same perceived direction.
                    horizontal_error = 0.5 - smoothed_x
                    vertical_error = 0.5 - smoothed_y

                    left_right_speed = speed_from_error(
                        horizontal_error,
                        DEAD_ZONE_X,
                        MAX_LEFT_RIGHT_SPEED,
                    )

                    up_down_speed = speed_from_error(
                        vertical_error,
                        DEAD_ZONE_Y,
                        MAX_UP_DOWN_SPEED,
                    )

                    directions = []

                    if left_right_speed < 0:
                        directions.append("LEFT")
                    elif left_right_speed > 0:
                        directions.append("RIGHT")

                    if up_down_speed > 0:
                        directions.append("UP")
                    elif up_down_speed < 0:
                        directions.append("DOWN")

                    if directions:
                        status = " + ".join(directions)
                    else:
                        status = "HOVER - PALM CENTRED"

                else:
                    depth_candidate = None
                    depth_candidate_started = None
                    depth_motion_started = None
                    status = "HOVER - GESTURE UNCLEAR"

            else:
                activation_started = None
                depth_candidate = None
                depth_candidate_started = None
                depth_motion_started = None
                depth_reset_started = None

                seconds_missing = now - last_hand_seen

                seconds_remaining = max(
                    0,
                    NO_HAND_LAND_SECONDS - seconds_missing,
                )

                status = (
                    f"HAND LOST - HOVERING - "
                    f"LAND IN {seconds_remaining:.0f}s"
                )

                if seconds_missing >= NO_HAND_LAND_SECONDS:
                    landing_reason = (
                        f"no hand detected for "
                        f"{NO_HAND_LAND_SECONDS} seconds"
                    )
                    break

            # Send continuous control commands.
            if now - last_rc_sent >= RC_INTERVAL:
                tello.send_rc_control(
                    left_right_speed,
                    forward_backward_speed,
                    up_down_speed,
                    0,
                )
                last_rc_sent = now

            # Check the battery every five seconds.
            if now - last_battery_check >= 5:
                battery = tello.get_battery()
                last_battery_check = now

                if battery <= LAND_AT_BATTERY:
                    landing_reason = (
                        f"battery reached {battery}%"
                    )
                    break

            cv2.putText(
                display,
                status,
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Gesture: {gesture_text}",
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Left/right: {left_right_speed}",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Forward/backward: {forward_backward_speed}",
                (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Up/down: {up_down_speed}",
                (25, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Battery: {battery}%",
                (25, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            current_progress = max(
                activation_progress,
                depth_progress,
                reset_progress,
            )

            if current_progress > 0:
                bar_width = int(
                    (width - 50) * current_progress
                )

                cv2.rectangle(
                    display,
                    (25, height - 60),
                    (width - 25, height - 42),
                    (80, 80, 80),
                    2,
                )

                cv2.rectangle(
                    display,
                    (25, height - 60),
                    (25 + bar_width, height - 42),
                    (0, 255, 255),
                    -1,
                )

            cv2.putText(
                display,
                "L or Q = land",
                (25, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "PalmPilot Gesture Tracking",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("l"), ord("L")):
                landing_reason = "L key pressed"
                break

            if key in (ord("q"), ord("Q")):
                landing_reason = "Q key pressed"
                break

    except KeyboardInterrupt:
        landing_reason = "program interrupted"

    except Exception as error:
        landing_reason = f"error: {error}"
        print(f"Error: {error}")

    finally:
        if flying:
            print(
                f"Landing: "
                f"{landing_reason or 'program finished'}"
            )

            try:
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.2)
                tello.land()
            except Exception as landing_error:
                print(f"Landing error: {landing_error}")

        if streaming:
            try:
                tello.streamoff()
            except Exception:
                pass

        try:
            tello.end()
        except Exception:
            pass

        recognizer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
