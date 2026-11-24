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
| `python -m gesture_control.filters`     | noisy vs. One Euro filtered signal             |
| `python -m gesture_control.mouse`       | cursor visits all four screen corners          |
| `python -m gesture_control.controller`  | prints ON/OFF when Ctrl+Alt+G is pressed       |
| `python -m gesture_control.main`        | the full app                                   |

On first run the MediaPipe hand model (`hand_landmarker.task`, ~8 MB) is downloaded from
Google's official model storage into `models/`. After that everything works offline.

## Stage 1 — how to verify

1. `python -m gesture_control.main` — a preview window opens with a blue work-zone rectangle.
2. Raise a hand so the index fingertip (red dot) is inside the rectangle.
   The cursor follows the finger smoothly, without jitter when the hand is still.
3. Move the fingertip to each corner of the rectangle — the cursor reaches every screen corner.
4. Press **Ctrl+Alt+G** — the status turns red (`Control: OFF`) and the cursor stops instantly,
   so the real mouse can be used. Press again to re-enable.
5. Check that the preview shows at least ~25 FPS.
6. Press `q` or `Esc` in the preview window to quit.

Tuning (all in `src/gesture_control/config.py`):

- the cursor is jittery → lower `FILTER_MIN_CUTOFF`;
- the cursor lags behind fast moves → raise `FILTER_BETA`;
- hard to reach the edges / arm gets tired → shrink the `WORK_ZONE_*` rectangle or raise `CURSOR_SENSITIVITY`.
