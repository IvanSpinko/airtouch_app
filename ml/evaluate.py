"""Evaluate the trained gesture classifier on the held-out test set.

Run from the project root: ``python -m ml.evaluate``

Uses the numpy runtime model (exactly what the app runs) and prints accuracy,
per-gesture precision / recall and the confusion matrix, plus whether the stage 3
targets (>= 500 samples per gesture, >= 95% accuracy) are met.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gesture_control import config
from gesture_control.gestures import GestureClassifier
from ml.dataset import load


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def report(cm: np.ndarray, names: list[str]) -> str:
    total = cm.sum()
    acc = np.trace(cm) / total if total else float("nan")
    width = max(len(n) for n in names) + 2
    lines = [f"accuracy: {acc:.4f}  ({np.trace(cm)}/{total})", ""]
    lines.append(f"{'gesture':<{width}}{'precision':>10}{'recall':>10}{'support':>10}")
    for i, name in enumerate(names):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else float("nan")
        rec = tp / cm[i].sum() if cm[i].sum() else float("nan")
        lines.append(f"{name:<{width}}{prec:>10.3f}{rec:>10.3f}{cm[i].sum():>10d}")
    lines += ["", "confusion matrix (rows = true, columns = predicted):"]
    short = [n[:6] for n in names]
    lines.append(" " * width + "".join(f"{s:>8}" for s in short))
    for i, name in enumerate(names):
        lines.append(f"{name:<{width}}" + "".join(f"{v:>8d}" for v in cm[i]))
    return "\n".join(lines)


def evaluate(
    model_path: Path = config.GESTURE_MODEL_NPZ, split_path: Path = config.GESTURE_TEST_SPLIT
) -> tuple[float, np.ndarray]:
    clf = GestureClassifier(model_path)
    data = np.load(split_path)
    probs = clf.predict_proba(data["features"])
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(data["labels"], pred, len(config.GESTURE_CLASSES))
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")
    return acc, cm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=config.GESTURE_MODEL_NPZ)
    parser.add_argument("--split", type=Path, default=config.GESTURE_TEST_SPLIT)
    args = parser.parse_args()
    if not args.model.exists() or not args.split.exists():
        raise SystemExit("No trained model yet. Run: python -m ml.train")

    acc, cm = evaluate(args.model, args.split)
    print(report(cm, list(config.GESTURE_CLASSES)))

    counts = load().counts()
    enough = all(n >= config.RECORD_TARGET_PER_CLASS for n in counts.values())
    print()
    print(f"samples per gesture: {counts}")
    print(f"[{'OK' if enough else 'TODO'}] >= {config.RECORD_TARGET_PER_CLASS} samples per gesture")
    print(f"[{'OK' if acc >= config.ML_TARGET_ACCURACY else 'TODO'}] "
          f"accuracy {acc:.4f} >= {config.ML_TARGET_ACCURACY}")


if __name__ == "__main__":
    main()
