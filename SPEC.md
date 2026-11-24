# Spec: Controlling a PC with Hand Gestures

## Goal

A Windows desktop application that tracks a hand through a webcam and lets the user control
the PC without a mouse or keyboard: move the cursor, click, scroll and type on the on-screen
keyboard using gestures.

- **Platform:** Windows 10/11, a regular webcam.
- **User:** one person in front of the computer, one hand (left or right).
- **ML part of the project:** a custom-trained gesture classifier on top of MediaPipe hand landmarks.

## Functional requirements

**Mouse.** The gestures below are tentative and may change after testing.

| Action          | Gesture                       |
|-----------------|-------------------------------|
| Move cursor     | Index fingertip               |
| Left click      | Pinch: thumb + index          |
| Double click    | Two quick pinches             |
| Right click     | Pinch: thumb + middle         |
| Drag            | Hold the pinch and move hand  |
| Scroll          | Two fingers up / down         |
| Pause tracking  | Fist                          |

**On-screen keyboard.**

1. The built-in Windows on-screen keyboard (`osk.exe`) is used. A custom keyboard is a later improvement.
2. It is opened and closed by a dedicated gesture (e.g. an open palm held for 1 s): the app
   launches `osk.exe` or terminates its process.
3. Typing uses the same virtual mouse: hover the cursor over a key and pinch.
4. The keyboard window is enlarged to make keys easier to hit. Word prediction is enabled in its options.
5. Layouts and language switching come from Windows.
6. The very first step of stage 4 is to verify that `pynput` clicks reach `osk.exe`.

**General.**

- Works with both the left and the right hand.
- A small indicator window: whether the camera sees the hand and which gesture is recognized.
- Calibration of the work zone and cursor sensitivity.
- Enable/disable via a hotkey and from the system tray.

## Non-functional requirements

The target values below are tentative and should be refined after the first prototype.

| Requirement                   | Target                        |
|-------------------------------|-------------------------------|
| Gesture-to-action latency     | under 100 ms                  |
| Frame processing rate         | at least 25 FPS               |
| Gesture classifier accuracy   | at least 95% on the test set  |
| False clicks                  | at most 1 per 5 minutes       |
| Typing speed                  | at least 10 words per minute  |
| CPU load                      | under 20%                     |

- **Privacy:** video is processed locally only; nothing is stored or sent over the network.
- **Conditions:** stable operation under normal room lighting, hand 0.3–1 m from the camera.
- **Ergonomics:** a small work zone so the hand does not get tired.
- **Limitation:** windows running with administrator rights (UAC) cannot be clicked — a Windows restriction.

## Technologies and architecture

Everything runs locally in Python; no cloud is required.

```
Webcam (OpenCV) -> MediaPipe Hands (21 landmarks) -> Classifier (own model) -> Controller (mouse / keyboard)
                                                          ^                          |             |
                     Offline: record gestures to CSV -> train (PyTorch)              v             v
                                                                              PyQt overlay       pynput
                                                                                     \             /
                                                                                 Windows: active window
```

The controller decides what to do with a gesture: in mouse mode it moves the cursor and clicks;
in keyboard mode it opens osk.exe, and typing is done with the same mouse.

| Component              | Technology                                      |
|------------------------|-------------------------------------------------|
| Language               | Python 3.11+                                    |
| Video capture          | OpenCV                                          |
| Hand tracking          | MediaPipe Hands                                 |
| Gesture classifier     | PyTorch (MLP, later LSTM for dynamic gestures)  |
| Cursor smoothing       | One Euro Filter                                 |
| Input control          | pynput                                          |
| Indicator and settings | PyQt6                                           |
| Packaging              | PyInstaller → `.exe`                            |
| Repository             | GitHub                                          |

## Milestones and acceptance criteria

Each stage ends with a working version that can be demonstrated.

- [x] **1. Tracking and cursor.** Done when the cursor smoothly follows the finger, reaches all
  screen corners, and a hotkey instantly disables control.
- [ ] **2. Clicks, drag, scroll.** Done when you can open a file, drag a window and scroll a page
  without a mouse.
- [ ] **3. Dataset and gesture classifier.** Done when at least 500 samples per gesture are
  collected and the model reaches at least 95% accuracy.
- [ ] **4. Keyboard via osk.exe.** Done when a gesture opens and closes the Windows on-screen
  keyboard and you can type and send a message using gestures only. Measure typing speed (WPM).
- [ ] **5. Settings and packaging.** Done when there is calibration, a tray icon and an `.exe`
  that runs on a clean Windows install.

Later, optionally: a custom PyQt6 keyboard with large keys and cursor snapping (compare typing
speed with osk.exe), swipe typing, an iPhone as a camera/remote via the cloud (AWS WebSocket).

## Development

### Repository layout

```
airtouch_app/
├── SPEC.md               # this spec
├── requirements.txt
├── src/gesture_control/
│   ├── main.py           # entry point, main loop
│   ├── camera.py         # frame capture (OpenCV)
│   ├── tracker.py        # MediaPipe Hands -> 21 landmarks
│   ├── filters.py        # One Euro Filter
│   ├── gestures.py       # threshold-based pinches + trained classifier
│   ├── controller.py     # mouse / keyboard / pause modes
│   ├── mouse.py          # pynput: cursor, clicks, scroll
│   ├── keyboard.py       # launching and closing osk.exe
│   ├── overlay.py        # PyQt6: indicator and settings
│   └── config.py         # all thresholds and parameters
├── ml/
│   ├── record.py         # record gestures to data/*.csv
│   ├── train.py          # train MLP -> models/gesture.pt
│   └── evaluate.py       # accuracy, confusion matrix
├── data/
├── models/
└── tests/
```

### Code rules

- Python 3.11+, virtual environment `.venv`, all dependencies in `requirements.txt`.
- All thresholds and settings live only in `config.py` — no magic numbers in code.
- From stage 1 on, the Ctrl+Alt+G hotkey immediately disables mouse control.
- Every module can be run on its own for checking, e.g. `python -m gesture_control.tracker`
  shows a window with hand landmarks.
- pytest tests for filters, gestures and the controller, without a camera: on recorded landmarks.
- Work follows the stages of this spec; after each one — stop and describe how to verify it. Small commits.
- All code, comments, docs and commit messages are in English.

### Commands (Windows)

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m gesture_control.main
pytest
```
