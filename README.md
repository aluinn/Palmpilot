# PalmPilot

PalmPilot is a Python computer-vision project that controls a Ryze Tello drone using hand gestures detected through its live camera.

## Features

- Live Tello camera feed
- MediaPipe hand and gesture recognition
- Continuous left, right, up and down tracking
- Forward and backward gestures
- Smoothed movement to reduce jitter
- Battery and hand-loss safety controls
- Automatic hovering and landing

## Controls

| Input | Action |
|---|---|
| Open palm ✋ | Control left, right, up and down |
| Victory sign ✌️ or Pointing Up ☝️ | Move forward |
| Closed fist ✊ | Move backward |
| Open palm in centre | Hover/reset depth movement |
| `L` or `Q` | Land |

Forward and backward movement is limited to two seconds before requiring an open-palm reset.

## Technology

- Python 3.12
- OpenCV
- MediaPipe
- DJITelloPy
- Ryze Tello SDK

## Installation

Clone the project:

```bash
git clone https://github.com/Aluinn/palmpilot.git
cd palmpilot
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the MediaPipe gesture model:

```bash
curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

## Running the Project

1. Charge the Tello and attach its propeller guards.
2. Connect the computer to the `TELLO-` Wi-Fi network.
3. Run:

```bash
python palmpilot.py
```

4. Type `TAKEOFF` when prompted.
5. Hold an open palm inside the yellow box to activate tracking.

## Safety

- There is no minimum battery requirement for takeoff.
- The drone lands when the battery reaches 20%.
- Losing the hand causes immediate hovering.
- The drone lands after 30 seconds without detecting a hand.
- Forward and backward movement uses a reduced speed.
- Press `L` or `Q` to land manually.

Always test indoors in a clear area with propeller guards attached.

## Project Tests

The repository includes separate programs for testing:

- Drone connection
- Camera streaming
- Gesture recognition
- Takeoff and landing
- Movement previews
- Gesture-controlled flight

## Roadmap

- Gesture-controlled photographs
- Flight data recording
- Annotated video recording
- Hand-distance following
- Calibration and improved tracking
- No-flight preview mode

## Acknowledgements

Built using [DJITelloPy](https://github.com/damiafuentes/DJITelloPy), [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer) and [OpenCV](https://opencv.org/).
## Demonstration 



https://github.com/user-attachments/assets/5d2c14f0-d38a-4ebf-91dc-3201be7157b9










