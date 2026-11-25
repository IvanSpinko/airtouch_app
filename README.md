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

The app asks for administrator rights (UAC) so its clicks reach the on-screen keyboard, see
[Stage 4](#stage-4--on-screen-keyboard); `--no-admin` skips that. `--log` records a session for
[diagnosing false clicks](#diagnosing-false-clicks).

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
| `python -m gesture_control.gestures`    | recognized gesture live                        |
| `python -m gesture_control.filters`     | noisy vs. One Euro filtered signal             |
| `python -m gesture_control.mouse`       | cursor visits all four screen corners          |
| `python -m gesture_control.controller`  | prints ON/OFF when Ctrl+Alt+G is pressed       |
| `python -m gesture_control.main`        | the full app                                   |
| `python -m gesture_control.keyboard`    | open / close the on-screen keyboard            |
| `python -m gesture_control.typing_test` | typing speed test (words per minute)           |
| `python -m ml.migrate`                  | reuse still-valid old recordings               |
| `python -m ml.record`                   | record gesture samples for training            |
| `python -m ml.train`                    | train the gesture classifier                   |
| `python -m ml.evaluate`                 | accuracy and confusion matrix on the test set  |
| `python -m ml.replay`                   | replay a `--log` session through the code      |

On first run the MediaPipe hand model (`hand_landmarker.task`, ~8 MB) is downloaded from
Google's official model storage into `models/`. After that everything works offline.

## Gestures

The gestures use big finger movements, which MediaPipe tracks reliably, instead of fingertip
contact. Both hands work (left hands are mirrored onto right ones).

| Action                 | Gesture                                                                   |
|------------------------|---------------------------------------------------------------------------|
| Move cursor            | **L pose**: index + thumb straight, middle / ring / pinky folded          |
| Left click             | from L, **bend the thumb** in and straighten it again                     |
| Double click           | two quick thumb bends                                                     |
| Drag                   | keep the thumb bent ~0.4 s (`DRAG READY`), move the hand, straighten to drop |
| Right click            | from L, **bend the index finger** and straighten it again                 |
| Scroll                 | index + middle **together**, thumb tucked; hold the hand above / below the start point |
| Pause / resume         | hold a fist ~1 s                                                          |
| On-screen keyboard     | hold a **shaka** 🤙 (only thumb + pinky out) ~1 s                          |
| Rest                   | any other pose (e.g. open palm) — the cursor stays where it is            |

The red dot in the preview is the cursor point: the base of the index finger. Neither the thumb
nor the index bend moves it, so clicks do not shift the cursor.

### How clicks work

- The cursor moves **only in the L pose**; every other pose is like letting go of a mouse.
- A click fires when the hand **returns to the L pose** after a bend. So closing the hand into a
  fist — which passes through both bent poses — never clicks, and clicks are also ignored for
  `AFTER_FIST_CLICK_GUARD_S` after a fist.
- A quick bend never holds the button down, so a false bend cannot drag or select. Dragging
  needs the thumb kept bent until `DRAG READY` flashes.
- While a bend is building up, the cursor stops, so the click lands where you aimed.
- The `L` / `R` bars at the bottom of the preview show how strongly the thumb / index bend is
  seen; the pose counts above the yellow line, the cursor stops above the orange one. The last
  action (`LEFT CLICK`, `DRAG READY`, `RIGHT CLICK`, `KEYBOARD`, `PAUSED`...) flashes at the top.
- With a trained model its probabilities are averaged over the last frames
  (`CLASSIFIER_SMOOTHING`), so one odd frame cannot trigger anything.

### Cursor

By default the cursor works like a mouse with pointer acceleration (`CURSOR_MODE = "relative"`):
slow hand movement for precise aiming, a quick flick to cross the screen; lowering the hand and
raising it elsewhere does not move the cursor. Single-frame landmark glitches are ignored
(`OUTLIER_SPEED`). `CURSOR_MODE = "absolute"` maps a work-zone rectangle onto the screen instead.

Tuning in `src/gesture_control/config.py`:

- jittery → lower `FILTER_MIN_CUTOFF`, raise `CURSOR_GLIDE_TAU_S`;
- lags behind fast moves → raise `FILTER_BETA`, lower `CURSOR_GLIDE_TAU_S`;
- too sensitive when aiming → lower `ACCEL_MIN_GAIN`; too slow to cross the screen → raise `ACCEL_MAX_GAIN`;
- the hand is lost often → more light, keep the whole hand in the frame (a red bar on a frame
  edge warns that it is leaving).

### Diagnosing false clicks

1. Run `python -m gesture_control.main --log` and use the app as usual.
2. Right after a false click or wrong gesture press **`m`** in the preview window (`MARKED`).
3. Quit, then run `python -m ml.replay`: it replays the newest `data/session-*.csv` through the
   current code and prints, for every mark, the last 3 seconds frame by frame (gesture, L / R
   strength, what the model saw, which action fired).

The log holds hand landmarks and actions only — no images — and stays in `data/` (not in git).

## Stage 1 — how to verify

1. `python -m gesture_control.main` — a preview window opens.
2. Show the L pose: the cursor follows smoothly, still when the hand is still.
3. Reach every screen corner: slow moves for aiming, quick flicks to travel far.
4. **Ctrl+Alt+G** — `Control: OFF`, the cursor stops instantly; press again to re-enable.
5. The preview shows at least ~25 FPS. `q` or `Esc` quits.

## Stage 2 — how to verify

1. **Open a file:** point at a file in Explorer, two quick thumb bends.
2. **Drag a window:** point at its title bar, keep the thumb bent until `DRAG READY`, move, straighten.
3. **Scroll:** on a long page show two fingers together, raise the hand a little and hold — the
   page keeps scrolling (orange line = start point); further = faster; change pose to stop.
4. **Right click:** bend the index finger and straighten it on the desktop — context menu.
5. **Pause:** hold a fist ~1 s (`PAUSED`), again to resume.

## Stage 3 — gesture classifier

### 1. Record data

The gesture set changed in stage 4, so the dataset is `data/gestures_v2.csv`. First reuse the
old recordings of the poses that did not change (scroll, fist, open palm):

```powershell
python -m ml.migrate
python -m ml.record
```

In the recorder press `1`…`8` to choose a gesture, `Space` to start / stop, `u` to undo the last
samples, `q` to quit. Only landmark coordinates are saved, never images. Goal: **500+ samples
per gesture** (the counter turns green).

| Key | Gesture       | How to record it                                                        |
|-----|---------------|-------------------------------------------------------------------------|
| 1   | `none`        | relaxed / half-open hand, transitions — anything that is not a gesture   |
| 2   | `l_pose`      | index + thumb straight, other fingers folded                             |
| 3   | `thumb_bent`  | L pose with the thumb bent in (index stays straight)                     |
| 4   | `index_bent`  | L pose with the index finger bent (thumb stays straight)                 |
| 5   | `two_fingers` | index + middle together, thumb tucked (migrated)                         |
| 6   | `fist`        | closed fist, thumb over the fingers (migrated)                           |
| 7   | `open_palm`   | all fingers spread (migrated)                                            |
| 8   | `shaka`       | only thumb + pinky out                                                   |

Record both hands, move and rotate the hand slowly, change the distance (0.3–1 m). For
`thumb_bent` / `index_bent` include light and full bends; for `l_pose` include a thumb that is
only slightly open, so the model learns where the boundary is.

### 2. Train and evaluate

```powershell
python -m ml.train
python -m ml.evaluate
```

`evaluate` prints accuracy, per-gesture precision / recall, the confusion matrix and whether the
stage targets are met (>= 500 samples per gesture, >= 95% accuracy).

### 3. Use it

The app picks up `models/gesture.npz` automatically (console: `trained classifier + rules`). A
model trained for an older gesture set is ignored with a message. `USE_CLASSIFIER = False`
uses the rules only.

## Stage 4 — on-screen keyboard

The Windows on-screen keyboard (`osk.exe`) runs with **high privileges**, and Windows blocks
clicks, window messages and termination coming from normal programs (UIPI) — verified at the
start of the stage. Therefore **AirTouch asks for administrator rights on start** (UAC) and
restarts itself elevated in a new console window. This also lifts the "cannot click
administrator windows" limitation.

- **Open / close:** hold a **shaka** 🤙 ~1 s — `KEYBOARD` flashes, the status shows `Keyboard`.
- **Type:** L pose over a key, quick thumb bend.
- The keyboard opens **enlarged** at the bottom of the screen with **word prediction** on;
  layouts and language switching are the usual Windows ones.

### How to verify

1. **Click check** from an **administrator** PowerShell in the project folder, without touching
   the mouse: `.venv\Scripts\python -m gesture_control.keyboard --check` must print
   `running as administrator: True` and `OK: pynput clicks reach osk.exe`.
2. Start the app, accept UAC; hold a shaka ~1 s — the keyboard opens.
3. In Notepad or a messenger type a message with thumb bends and press the keyboard's Enter.
4. Shaka again — the keyboard closes.
5. **Typing speed:** `python -m gesture_control.typing_test` in another terminal; for each
   phrase press Enter, type it with gestures, press Enter. Target: at least 10 WPM.

In PyCharm the elevated copy starts in its own console window and PyCharm's run ends right
away; start PyCharm as administrator to keep everything in its console.
