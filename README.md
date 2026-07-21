# PalmPilot

**PalmPilot** is a Python computer-vision project that uses hand gestures to interact with a **Ryze Tello drone**.

The project receives the Tello's live camera stream, detects a hand using **MediaPipe**, draws its landmarks and identifies gestures in real time.

> **Note:** Flight control is not enabled yet. The current version focuses on safe connection, video streaming and gesture recognition.

## Current Features

- Connects to a Tello over Wi-Fi
- Reads and displays the battery level
- Displays the live Tello camera feed
- Detects 21 hand landmarks
- Recognises gestures such as:
  - Open palm
  - Closed fist
  - Thumbs up
  - Thumbs down
  - Victory sign
  - Pointing up
- Cleans up the video stream when the program closes
- Preview-only gesture controller
- Gesture hold confirmation to prevent accidental commands
- Disarmed, armed and simulated flying states
- Battery-checked takeoff, hover and automatic landing test
- Safety landing if the flight program is interrupted

## Technology

- **Python 3.12**
- **DJITelloPy**
- **OpenCV**
- **MediaPipe Gesture Recognizer**
- **Git and GitHub**

## Project Files

- `connect_test.py` — tests the connection and reads the battery level
- `camera_test.py` — displays the live camera stream
- `gesture_test.py` — detects hands and recognises gestures
- `requirements.txt` — lists the Python dependencies
- `controller_preview.py` — safely previews gesture-controlled flight states without starting the motors
- `flight_test.py` — performs a confirmed three-second takeoff, hover and landing test

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the MediaPipe gesture-recognition model into the project folder:

```bash
curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

## Running the Project

1. Fully charge the Tello battery.
2. Close the Tello mobile app.
3. Disconnect the phone from the Tello.
4. Switch on the Tello.
5. Connect the computer to the `TELLO-XXXXXX` Wi-Fi network.
6. Run one of the test programs:

```bash
python connect_test.py
python camera_test.py
python gesture_test.py
```

When using the camera or gesture test, click the video window and press **Q** to close it.

## Safety

The current version does **not** issue flight commands.

Future flight controls will include:

- Deliberate gesture confirmation
- Battery checks
- Command cooldowns
- An immediate keyboard-controlled landing option

Testing should only take place in a clear, safe flying area with propeller guards fitted.

## Roadmap

- [x] Connect Python to the Tello
- [x] Read the battery level
- [x] Display the camera stream
- [x] Detect hand landmarks
- [x] Recognise hand gestures
- [x] Add stable gesture confirmation
- [x] Add a preview-only control mode
- [x] Add arming and disarming
- [x] Complete a controlled takeoff-and-landing test
- [x] Connect confirmed gestures to takeoff and landing
- [x] Add safe takeoff and landing controls
- [ ] Add hand-position movement
- [ ] Add an emergency keyboard landing command
- [ ] Record a demonstration video

## Status

**Work in progress:** Connection, camera streaming and hand-gesture recognition are working.

## Demo of hand tracking


https://github.com/user-attachments/assets/d05adb9c-1b01-4fe1-9f7b-dd9586b74169




