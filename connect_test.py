from djitellopy import Tello

tello = Tello()

try:
    print("Connecting to Tello...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Connected successfully! Battery: {battery}%")

finally:
    tello.end()

    