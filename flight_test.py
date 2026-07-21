import time

from djitellopy import Tello


MINIMUM_BATTERY = 50
HOVER_SECONDS = 3

tello = Tello()
takeoff_attempted = False

try:
    print("Connecting to Tello...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    if battery < MINIMUM_BATTERY:
        print(f"Battery must be at least {MINIMUM_BATTERY}%.")
        raise SystemExit

    confirmation = input(
        "Type TAKEOFF to begin, or anything else to cancel: "
    )

    if confirmation != "TAKEOFF":
        print("Flight cancelled.")
        raise SystemExit

    print("Takeoff in:")

    for number in range(3, 0, -1):
        print(number)
        time.sleep(1)

    # Set this before sending the command so the safety
    # cleanup attempts to land even if takeoff reports an error.
    takeoff_attempted = True

    print("Taking off...")
    tello.takeoff()

    print(f"Hovering for {HOVER_SECONDS} seconds...")
    time.sleep(HOVER_SECONDS)

    print("Landing...")
    tello.land()
    takeoff_attempted = False

    print("Flight test completed successfully.")

except KeyboardInterrupt:
    print("\nFlight interrupted.")

finally:
    if takeoff_attempted:
        print("Safety landing...")
        try:
            tello.land()
        except Exception as error:
            print(f"Landing command failed: {error}")

    tello.end()
    print("Connection closed.")
    