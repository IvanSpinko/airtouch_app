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

**On-screen keyboard (stage 4).**

1. The built-in Windows on-screen keyboard (`osk.exe`) is used. It is replaced by a custom
   keyboard in stage 6 and stays available as an option.
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

**Gesture set (revised in stage 4).** Testing showed that fingertip pinches are the least
reliable thing MediaPipe tracks, so the mouse gestures were changed to big finger movements:

| Action          | Gesture                                                     |
|-----------------|-------------------------------------------------------------|
| Move cursor     | L pose: index + thumb straight, other fingers folded        |
| Left click      | bend the thumb and return to L (hold = drag)                |
| Right click     | bend the index finger and return to L                       |
| Scroll          | index + middle together, hand above / below the start point |
| Pause           | fist                                                        |
| Keyboard        | shaka (thumb + pinky) held 1 s                              |

**Custom on-screen keyboard (stage 6).** `osk.exe` is built for a mouse, which kept typing
slow (4.3 words per minute on the first attempt), and it only accepts input from an app
running as administrator. A keyboard designed for gestures replaces it:

1. **Own window (PyQt6):** always on top at the bottom of the screen, large keys (sizes in
   `config.py`). It **never takes the keyboard focus**, so typed text goes to the app the user
   is working in.
2. **Open / close** with the same gesture as before (shaka held 1 s). A setting chooses the
   keyboard: custom (default) or `osk.exe`.
3. **Typing:** in the L pose the key under the cursor is highlighted; bending the thumb presses
   it. The key press is sent directly (not as a simulated mouse click on a button).
4. **Cursor snapping:** over the keyboard the cursor snaps to the center of the nearest key,
   with hysteresis (it moves to a neighbouring key only when clearly closer to it), so keys
   are hit without precise aiming.
5. **Keys:** letters, digits, basic punctuation, Space, Backspace, Enter, Shift, language switch.
   **English and Russian layouts** are part of the keyboard itself; characters are sent as
   Unicode, so the Windows input language does not matter.
6. **Word suggestions:** a row of 3 suggested words for the word being typed, from a word
   frequency list of the current language; choosing one inserts the word and a space.
7. **No administrator rights** are needed with the custom keyboard (the UAC prompt is only
   used when `osk.exe` is selected).

## Non-functional requirements

The target values below are tentative and should be refined after the first prototype.

| Requirement                   | Target                        |
|-------------------------------|-------------------------------|
| Gesture-to-action latency     | under 100 ms                  |
| Frame processing rate         | at least 25 FPS               |
| Gesture classifier accuracy   | at least 95% on the test set  |
| False clicks                  | at most 1 per 5 minutes       |
| Typing speed                  | at least 10 words per minute after practice (refined, see below) |
| CPU load                      | under 20%                     |

- **Privacy:** video is processed locally only; nothing is stored or sent over the network.
- **Conditions:** stable operation under normal room lighting, hand 0.3–1 m from the camera.
- **Ergonomics:** a small work zone so the hand does not get tired.
- **Limitation:** windows running with administrator rights (UAC) cannot be clicked — a Windows
  restriction. AirTouch therefore runs as administrator (needed for `osk.exe` anyway), which
  lifts this limitation.

**Typing speed — refined after measuring (stage 4).** First measured attempt with the
on-screen keyboard and gestures, without practice (`python -m gesture_control.typing_test`,
3 phrases): **4.3 words per minute (net), 94% accuracy**. Typing speed depends strongly on
how used the person is to this input method, and `osk.exe` is designed for a mouse, not for
gestures. The target is therefore refined to **at least 10 words per minute after practice**;
the baseline above is kept for comparison with the custom keyboard of stage 6.

## Technologies and architecture

Everything runs locally in Python; no cloud is required. The tray app runs the processing
loop (engine) in a background thread and the PyQt6 interface in the main thread.

```
Webcam (OpenCV) -> MediaPipe Hands (21 landmarks) -> Classifier (own model) -> Controller (mouse / keyboard)
                                                          ^                          |             |
                     Offline: record gestures to CSV -> train (PyTorch)              v             v
                                                                              PyQt overlay       pynput
                                                                                     \             /
                                                                                 Windows: active window
```

The controller decides what to do with a gesture: it moves the cursor, clicks, drags, scrolls,
pauses and opens the on-screen keyboard. With the custom keyboard (stage 6) the cursor snaps to
its keys and a thumb bend types the highlighted key.

| Component              | Technology                                      |
|------------------------|-------------------------------------------------|
| Language               | Python 3.11+                                    |
| Video capture          | OpenCV                                          |
| Hand tracking          | MediaPipe Hands                                 |
| Gesture classifier     | PyTorch MLP (trained), numpy at runtime          |
| Cursor smoothing       | One Euro Filter                                 |
| Input control          | pynput                                          |
| Tray, indicator, settings, custom keyboard | PyQt6                       |
| Packaging              | PyInstaller → `.exe`                            |
| Repository             | GitHub                                          |

## Milestones and acceptance criteria

Each stage ends with a working version that can be demonstrated.

- [x] **1. Tracking and cursor.** Done when the cursor smoothly follows the finger, reaches all
  screen corners, and a hotkey instantly disables control.
- [x] **2. Clicks, drag, scroll.** Done when you can open a file, drag a window and scroll a page
  without a mouse.
- [x] **3. Dataset and gesture classifier.** Done when at least 500 samples per gesture are
  collected and the model reaches at least 95% accuracy.
- [x] **4. Keyboard via osk.exe.** Done when a gesture opens and closes the Windows on-screen
  keyboard and you can type and send a message using gestures only. Measure typing speed (WPM).
  Measured: 4.3 WPM on the first attempt (target refined, see non-functional requirements).
- [x] **5. Settings and packaging.** Done when there is calibration, a tray icon and an `.exe`
  that runs on a clean Windows install.
- [ ] **6. Custom on-screen keyboard (last development stage).** Done when the shaka opens and
  closes the custom keyboard, you can type and send a message in English and in Russian using
  gestures only, the cursor snaps to keys, word suggestions work, the app no longer needs
  administrator rights with this keyboard, and typing speed is measured with
  `typing_test` and compared with the `osk.exe` baseline (4.3 WPM).
- [ ] **7. Finalization.** No new features. Done when the remaining non-functional
  requirements are measured and recorded (gesture-to-action latency, frame rate, CPU load),
  the small issues found during testing are fixed, README and SPEC describe the final state,
  and the final version is released on GitHub with the `.exe`.

**Out of scope** (considered, not planned for this project): swipe typing, an iPhone as a
camera / remote via the cloud (AWS WebSocket), an LSTM for dynamic gestures, Linux support.

## Development

### Repository layout

```
airtouch_app/
├── SPEC.md               # this spec
├── requirements.txt
├── src/gesture_control/
│   ├── app.py            # tray app (entry point of AirTouch.exe)
│   ├── main.py           # command-line entry point with the camera window
│   ├── engine.py         # processing loop: camera -> tracking -> gestures -> mouse
│   ├── camera.py         # frame capture (OpenCV)
│   ├── tracker.py        # MediaPipe Hands -> 21 landmarks
│   ├── filters.py        # One Euro Filter
│   ├── gestures.py       # gesture rules + trained classifier
│   ├── controller.py     # gestures -> cursor, clicks, drag, scroll, pause, keyboard
│   ├── mouse.py          # pynput: cursor, clicks, scroll; pointer acceleration
│   ├── keyboard.py       # launching and closing osk.exe
│   ├── onscreen.py       # custom on-screen keyboard: layouts, snapping, suggestions
│   ├── onscreen_window.py # PyQt6 window of the custom keyboard
│   ├── overlay.py        # PyQt6: status indicator, tray icons
│   ├── dialogs.py        # PyQt6: settings and calibration windows
│   ├── settings.py       # user settings (%APPDATA%\AirTouch\settings.json)
│   ├── calibration.py    # calibration math
│   ├── preview.py        # debug camera window drawing
│   ├── typing_test.py    # typing speed test (WPM)
│   ├── sessionlog.py     # optional session log for diagnosing false clicks
│   └── config.py         # all thresholds and parameters
├── ml/
│   ├── dataset.py        # CSV storage, features, splits, augmentation
│   ├── record.py         # record gestures to data/*.csv
│   ├── train.py          # train MLP -> models/gesture.pt
│   ├── migrate.py        # reuse recordings of the first gesture set
│   ├── evaluate.py       # accuracy, confusion matrix
│   └── replay.py         # replay a session log through the current code
├── scripts/build.py      # builds and self-tests AirTouch.exe
├── docs/releases/        # release notes
├── data/                 # recordings (local only, not in git)
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
python -m gesture_control.app      # tray app
python -m gesture_control.main     # with the camera window
pytest
python -m scripts.build            # AirTouch.exe
```
