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
