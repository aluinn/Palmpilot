import cv2
from djitellopy import Tello

tello = Tello()

try:
    print("Connecting...")
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    print("Starting camera...")
    tello.streamon()
    frame_reader = tello.get_frame_read()

    print("Camera running — press Q to close")

    while True:
        frame = frame_reader.frame
        cv2.imshow("Tello Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()