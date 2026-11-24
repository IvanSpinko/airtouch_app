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
the preview and stretched
over the whole screen and a hand position always maps to the same screen point.

## Gestures

| Action         | Gesture                                                      |
|----------------|--------------------------------------------------------------|
| Move cursor    | point with the index finger                                  |
| Left click     | quick pinch: thumb + **middle** finger                       |
| Double click   | two quick thumb + middle pinches                             |
| Drag           | thumb + middle pinch, hold ~0.3 s, then move; open to drop   |
| Right click    | pinch: thumb + index finger                                  |
| Scroll         | index + middle **held together**, thumb tucked, ring + pinky folded; then hold the hand above / below that spot (auto scroll) |
| Pause / resume | hold a fist for ~1 s                                         |

## Stage 2 — how to verify

1. `python -m gesture_control.main`; the preview shows the recognized `Gesture:`.
2. **Click / open a file:** point at a file in Explorer, make two quick thumb + middle
   pinches — it opens. The index finger stays still, so the cursor does not move.
3. **Drag a window:** point at a window title bar, thumb + middle pinch, move the hand, open.
4. **Scroll:** open a long web page, show two fingers held together, then raise the hand a
   little and hold it — the page keeps scrolling (orange line in the preview = start point);
   further from the line = faster; open the fingers to stop.
5. **Right click:** thumb + index pinch on the desktop opens the context menu.
6. **Pause:** hold a fist ~1 s — status turns `PAUSED`, nothing reacts; hold a fist again to resume.

Ready-to-click pose: index points, middle and thumb stick out ready to pinch, ring and
pinky folded. This is deliberately *not* scroll — scroll needs index and middle together.

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
