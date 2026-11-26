"""Build AirTouch.exe with PyInstaller and self-test it.

Run from the project root: ``python -m scripts.build`` (``--onedir`` for a folder
build that starts faster, ``--no-test`` to skip the self-test).

The .exe bundles the MediaPipe hand model and the trained gesture model, so it
works offline on a clean Windows. PyTorch and the ML tools are left out: the app
runs the gesture model with numpy.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gesture_control import config  # noqa: E402

BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist"
ICON_PATH = BUILD_DIR / "airtouch.ico"
ENTRY = ROOT / "scripts" / "airtouch_entry.py"
# Not needed at runtime. (matplotlib must stay: mediapipe imports it on load.)
EXCLUDE = ("torch", "torchvision", "ml", "pytest", "tkinter", "IPython")
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def make_icon(path: Path) -> Path:
    """Renders the app icon (same as the tray icon) into a multi-size .ico."""
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication

    from gesture_control.overlay import make_pixmap

    app = QApplication.instance() or QApplication([])  # noqa: F841 - needed for painting
    path.parent.mkdir(parents=True, exist_ok=True)
    biggest = make_pixmap(config.STATUS_COLOR_ON, max(ICON_SIZES)).toImage()
    if not biggest.save(str(path), "ICO"):
        # Qt without the ICO plugin: fall back to PNG frames packed by Pillow
        from PIL import Image

        png = path.with_suffix(".png")
        biggest.convertToFormat(QImage.Format.Format_ARGB32).save(str(png), "PNG")
        Image.open(png).save(path, sizes=[(s, s) for s in ICON_SIZES])
    return path


def pyinstaller_args(onedir: bool) -> list[str]:
    models = {
        config.HAND_MODEL_PATH: "models",
        config.GESTURE_MODEL_NPZ: "models",
        **{path: "models" for path in config.WORDS_PATHS.values()},
    }
    missing = [str(p) for p in models if not p.exists()]
    if missing:
        raise SystemExit(f"Missing model files: {missing}. Run the app once / train first.")
    args = [
        str(ENTRY),
        "--name", config.APP_NAME,
        "--windowed",
        "--noconfirm",
        "--clean",
        "--onedir" if onedir else "--onefile",
        "--icon", str(make_icon(ICON_PATH)),
        "--paths", str(ROOT / "src"),
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR / "pyinstaller"),
        "--specpath", str(BUILD_DIR),
        "--collect-all", "mediapipe",
    ]
    for src, dest in models.items():
        args += ["--add-data", f"{src}{';'}{dest}"]
    for module in EXCLUDE:
        args += ["--exclude-module", module]
    return args


def exe_path(onedir: bool) -> Path:
    name = f"{config.APP_NAME}.exe"
    return DIST_DIR / config.APP_NAME / name if onedir else DIST_DIR / name


def self_test(exe: Path) -> bool:
    """Runs the built exe with --self-test (no camera, no mouse, no admin prompt)."""
    report_path = config.APP_DATA_DIR / config.SELF_TEST_FILE
    report_path.unlink(missing_ok=True)  # never read a stale report
    result = subprocess.run([str(exe), "--self-test", "--no-admin"], timeout=300)
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "no report (crashed?)"
    print(f"self-test exit code {result.returncode}: {report}")
    return result.returncode == 0 and report == "OK"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onedir", action="store_true", help="folder build instead of one file")
    parser.add_argument("--no-test", action="store_true", help="skip the self-test")
    args = parser.parse_args()

    import PyInstaller.__main__

    PyInstaller.__main__.run(pyinstaller_args(args.onedir))
    exe = exe_path(args.onedir)
    size_mb = sum(f.stat().st_size for f in exe.parent.rglob("*") if f.is_file()) / 2**20 \
        if args.onedir else exe.stat().st_size / 2**20
    print(f"\nbuilt {exe} ({size_mb:.0f} MB)")
    if not args.no_test and not self_test(exe):
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()
