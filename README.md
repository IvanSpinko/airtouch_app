# AirTouch

Control a Windows PC with hand gestures through a regular webcam: cursor, clicks, scrolling and
typing on the Windows on-screen keyboard. Everything runs locally — video never leaves the machine.

Full spec: [SPEC.md](SPEC.md).

## Setup (Windows)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`torch` is only needed to train the gesture model; the app runs it with numpy.

## Run

```powershell
python -m gesture_control.main
```

## Tests

```powershell
pytest
```

## Modules

Every module can be run on its own:

| Command                                 | What it shows                                  |
|-----------------------------------------|------------------------------------------------|
| `python -m gesture_control.camera`      | raw webcam video and FPS                       |
| `python -m gesture_control.tracker`     | hand skeleton (21 landmarks) and handedness    |
| `python -m gesture_control.gestures`    | recognized gesture and pinch distances live    |
| `python -m gesture_control.filters`     | noisy vs. One Euro filtered signal             |
| `python -m gesture_control.mouse`       | cursor visits all four screen corners          |
| `python -m gesture_control.controller`  | prints ON/OFF when Ctrl+Alt+G is pressed       |
| `python -m gesture_control.main`        | the full app                                   |
| `python -m ml.record`                   | record gesture samples for training            |
| `python -m ml.train`                    | train the gesture classifier                   |
| `python -m ml.evaluate`                 | accuracy and confusion matrix on the test set  |

On first run the MediaPipe hand model (`hand_landmarker.task`, ~8 MB) is downloaded from
Google's official model storage into `models/`. After that everything works offline.

## Stage 1 — how to verify

1. `python -m gesture_control.main` — a preview window opens.
2. Raise a hand; the index fingertip is marked with a red dot. Keep the whole hand in the
   frame — a red bar on a frame edge means the hand is about to leave it and get lost.
   The cursor follows the finger smoothly, without jitter when the hand is still.
3. Reach every screen corner: slow moves for aiming, quick flicks to travel far.
4. Press **Ctrl+Alt+G** — the status turns red (`Control: OFF`) and the cursor stops instantly,
   so the real mouse can be used. Press again to re-enable.
5. Check that the preview shows at least ~25 FPS.
6. Press `q` or `Esc` in the preview window to quit.

Tuning (all in `src/gesture_control/config.py`):

- the cursor is jittery → lower `FILTER_MIN_CUTOFF`;
- the cursor lags behind fast moves → raise `FILTER_BETA`;
- hard to reach the edges / arm gets tired → shrink the `WORK_ZONE_*` rectangle or raise `CURSOR_SENSITIVITY`.

## Cursor modes

By default the cursor works like a mouse with pointer acceleration (`CURSOR_MODE = "relative"`):

- **slow** hand movement moves the cursor slowly — for precise aiming at small targets;
- a **quick flick** sends it across the screen;
- lowering the hand and raising it somewhere else does not move the cursor (like lifting a mouse).

Tuning in `config.py`:

- still too sensitive when aiming → lower `ACCEL_MIN_GAIN` (e.g. `0.15`);
- too slow to cross the screen → raise `ACCEL_MAX_GAIN`;
- acceleration kicks in too early / late → `ACCEL_LOW_SPEED_PX_S` / `ACCEL_HIGH_SPEED_PX_S`.

`CURSOR_MODE = "absolute"` restores the old behaviour: a blue work-zone rectangle is shown in
the preview and stretched over the whole screen, so a hand position always maps to the same
screen point.

## Gestures

| Action         | Gesture                                                      |
|----------------|--------------------------------------------------------------|
| Move cursor    | point with the index finger                                  |
| Left click     | quick pinch: thumb + **index** finger                        |
| Double click   | two quick thumb + index pinches                              |
| Drag           | thumb + index pinch, hold ~0.3 s, then move; open to drop    |
| Right click    | pinch: thumb + **middle** finger                             |
| Scroll         | index + middle **held together**, thumb tucked, ring + pinky folded; then hold the hand above / below that spot (auto scroll) |
| Pause / resume | hold a fist for ~1 s                                         |

## Stage 2 — how to verify

1. `python -m gesture_control.main`; the preview shows the recognized `Gesture:`.
2. **Click / open a file:** point at a file in Explorer, make two quick thumb + index
   pinches — it opens. The cursor follows the index knuckle, so pinching does not move it.
3. **Drag a window:** point at a window title bar, thumb + index pinch, hold, move the hand, open.
4. **Scroll:** open a long web page, show two fingers held together, then raise the hand a
   little and hold it — the page keeps scrolling (orange line in the preview = start point);
   further from the line = faster; open the fingers to stop.
5. **Right click:** thumb + middle pinch on the desktop opens the context menu.
6. **Pause:** hold a fist ~1 s — status turns `PAUSED`, nothing reacts; hold a fist again to resume.

A hand getting ready to pinch is deliberately *not* scroll — scroll needs index and middle
together with the thumb tucked away from both.

If a gesture is not recognized, run `python -m gesture_control.gestures` and look at the
`pi` / `pm` pinch distances and the index-middle `gap`, then tune in `config.py`:

- clicks do not trigger → raise `PINCH_ON_RATIO`; accidental clicks → lower it;
- drag / selection when you only wanted to click → raise `DRAG_MIN_HOLD_S` or `DRAG_START_PX`;
- scroll does not start → raise `SCROLL_FINGERS_TOGETHER_RATIO` or lower `SCROLL_THUMB_AWAY_RATIO`;
- the page scrolls while getting ready to click → the opposite of the line above;
- double click is not registered → raise `CLICK_SETTLE_S`;
- auto scroll too slow / fast → `SCROLL_AUTO_SPEED`, `SCROLL_AUTO_MAX_TICKS_S`;
- scrolls while the hand is held still → raise `SCROLL_AUTO_DEADZONE`;
- direction → `SCROLL_INVERT`; page-follows-hand scrolling instead → `SCROLL_MODE = "drag"`;
- swap which pinch is left / right → `LEFT_BUTTON_GESTURE`, `RIGHT_BUTTON_GESTURE`;
- the hand is lost too often → lower `MIN_HAND_*_CONFIDENCE`, add light, keep the
  whole hand inside the camera frame;
- the cursor freezes while pointing → the preview shows `fist` by mistake:
  lower `FIST_TIP_RATIO` or raise `DEPTH_WEIGHT`.

## Cursor stabilization

- The cursor follows the **base knuckle of the index finger**, not the fingertip: it moves with
  the hand but ignores finger bending and pinches. Change with `CURSOR_POINT_WEIGHTS`
  (`{LM_INDEX_TIP: 1.0}` = fingertip, a blend = finer but less stable).
- **Glitch rejection:** single-frame landmark jumps are ignored; if the hand really is elsewhere
  the cursor re-anchors without jumping (`OUTLIER_SPEED`, `OUTLIER_CONFIRM_FRAMES`).
- **Click safety:** a pinch must hold for `PINCH_CONFIRM_FRAMES` frames, release is debounced
  (`PINCH_RELEASE_FRAMES`), no clicks for `ACQUIRE_GRACE_FRAMES` after the hand appears, and with a
  trained model a pinch only clicks if the model agrees (`CLASSIFIER_PINCH_MIN_PROB`).

## Stage 3 — gesture classifier

### 1. Record data

```powershell
python -m ml.record
```

Press `1`…`7` to choose a gesture, `Space` to start / stop recording, `u` to undo the last
samples, `q` to quit. Only landmark coordinates are saved (`data/gestures.csv`), never images.
The goal is **500+ samples per gesture** (the counter turns green).

| Key | Gesture        | How to record it                                                   |
|-----|----------------|--------------------------------------------------------------------|
| 1   | `none`         | relaxed hand, transitions, half-open poses — anything else         |
| 2   | `point`        | index finger out, others folded. **Also record "almost pinch":**   |
|     |                | thumb close to the index finger but not touching, slowly closing   |
|     |                | and opening — otherwise the model clicks on a slight squeeze       |
| 3   | `pinch_index`  | thumb touching index fingertip — record both with the other fingers |
|     |                | **curled into a fist** and with them **spread out** ("OK" sign)    |
| 4   | `pinch_middle` | thumb touching middle fingertip                                    |
| 5   | `two_fingers`  | index + middle together, thumb tucked                              |
| 6   | `fist`         | closed fist                                                        |
| 7   | `open_palm`    | all fingers spread                                                 |

Tips: while recording, slowly move and rotate the hand, change distance (0.3–1 m), use both
hands and record in two or three sessions with different lighting. Variety matters more
than count.

### 2. Train and evaluate

```powershell
python -m ml.train
python -m ml.evaluate
```

Training takes seconds and writes `models/gesture.npz`; `evaluate` prints accuracy, per-gesture
precision / recall, the confusion matrix and whether the stage targets are met
(>= 500 samples per gesture, >= 95% accuracy). If a pair of gestures is confused, record more
varied samples of both and retrain.

### 3. Use it

`python -m gesture_control.main` picks the model up automatically (console:
`trained classifier + rules`; the preview shows the model confidence next to the gesture).
Delete `models/gesture.npz` or set `USE_CLASSIFIER = False` to go back to rules only.
