"""Copy still-valid recordings from the first gesture set into the current dataset.

Run from the project root: ``python -m ml.migrate``

The first gesture set used pinches; its ``two_fingers``, ``fist`` and ``open_palm``
recordings show the same poses in the new set, so they are copied
(``MIGRATE_V1_CLASSES``). The old ``none`` is not: it contains thumbs-up poses,
which are ``index_bent`` now. The old file is left untouched. Running it twice
does not duplicate rows.
"""

from __future__ import annotations

import csv
from pathlib import Path

from gesture_control import config
from ml.dataset import HEADER


def migrate(
    src: Path = config.GESTURE_DATA_V1_PATH,
    dst: Path = config.GESTURE_DATA_PATH,
    classes: tuple[str, ...] = config.MIGRATE_V1_CLASSES,
) -> dict[str, int]:
    """Appends ``classes`` rows from ``src`` to ``dst``; returns rows copied per class."""
    if not src.exists():
        return {}
    existing: set[tuple[str, ...]] = set()
    if dst.exists():
        with dst.open(newline="", encoding="utf-8") as f:
            existing = {tuple(row) for row in csv.reader(f)}
    copied = {c: 0 for c in classes}
    with src.open(newline="", encoding="utf-8") as f_in:
        rows = [row for row in csv.reader(f_in)][1:]
    new = not dst.exists() or dst.stat().st_size == 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        if new:
            w.writerow(HEADER)
        for row in rows:
            if row and row[0] in classes and tuple(row) not in existing:
                w.writerow(row)
                copied[row[0]] += 1
    return copied


def main() -> None:
    copied = migrate()
    if not copied:
        print(f"No old recordings at {config.GESTURE_DATA_V1_PATH}")
        return
    print(f"Copied into {config.GESTURE_DATA_PATH.name}: {copied}")
    print("Now record the new gestures: python -m ml.record")


if __name__ == "__main__":
    main()
