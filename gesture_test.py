from pathlib import Path

import cv2
import mediapipe as mp
from djitellopy import Tello


MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

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

try:
    print("Connecting...")
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    stream_started = True
    frame_reader = tello.get_frame_read()

    print("Show your hand to the camera — press Q to close")

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
        label = "No hand detected"

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
                    (255, 0, 0),
                    2,
                )

            for point in points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

        if result.gestures and result.gestures[0]:
            gesture = result.gestures[0][0]
            label = f"{gesture.category_name}: {gesture.score:.0%}"

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("PalmPilot Gesture Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if stream_started:
        tello.streamoff()

    tello.end()
    recognizer.close()
    cv2.destroyAllWindows()