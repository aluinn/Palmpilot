import time

from djitellopy import Tello


MINIMUM_BATTERY = 50
MOVEMENT_DISTANCE = 20

tello = Tello()
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
        "Type MOVE_TEST to begin, or anything else to cancel: "
    )

    if confirmation != "MOVE_TEST":
        print("Movement test cancelled.")
        raise SystemExit

    print("Starting in:")

    for number in range(3, 0, -1):
        print(number)
        time.sleep(1)

    flight_may_be_active = True

    print("Taking off...")
    tello.takeoff()
    time.sleep(2)

    print("Moving left 20 cm...")
    tello.move_left(MOVEMENT_DISTANCE)
    time.sleep(2)

    print("Moving right 20 cm...")
    tello.move_right(MOVEMENT_DISTANCE)
    time.sleep(2)

    print("Landing...")
    tello.land()
    flight_may_be_active = False

    print("Movement test completed successfully.")

except KeyboardInterrupt:
    print("\nMovement test interrupted.")

finally:
    if flight_may_be_active:
        print("Safety landing...")

        try:
            tello.land()
        except Exception as error:
            print(f"Landing command failed: {error}")

    tello.end()
    print("Connection closed.")
