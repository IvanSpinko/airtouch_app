"""Train the gesture MLP on recorded landmarks.

Run from the project root: ``python -m ml.train``

Writes:
- ``models/gesture.pt``   — PyTorch checkpoint (state dict + metadata);
- ``models/gesture.npz``  — the same weights for numpy inference in the app;
- ``models/test_split.npz`` — the held-out test set for ``python -m ml.evaluate``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gesture_control import config
from ml.dataset import Dataset, augment, load, stratified_split

NUM_FEATURES = config.NUM_LANDMARKS * 3


def build_model(num_classes: int, hidden=config.ML_HIDDEN, dropout=config.ML_DROPOUT) -> nn.Module:
    layers: list[nn.Module] = []
    size = NUM_FEATURES
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU(), nn.Dropout(dropout)]
        size = h
    layers.append(nn.Linear(size, num_classes))
    return nn.Sequential(*layers)


def export_numpy(model: nn.Module, path: Path) -> None:
    """Saves Linear layers as W0, b0, W1, b1, ... for GestureClassifier."""
    linears = [m for m in model if isinstance(m, nn.Linear)]
    arrays = {"classes": np.asarray(config.GESTURE_CLASSES), "num_layers": np.asarray(len(linears))}
    for i, m in enumerate(linears):
        arrays[f"W{i}"] = m.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    if len(y) == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(1) == y).float().mean())


def train(
    ds: Dataset,
    epochs: int = config.ML_EPOCHS,
    out_pt: Path = config.GESTURE_MODEL_PT,
    out_npz: Path = config.GESTURE_MODEL_NPZ,
    out_test: Path = config.GESTURE_TEST_SPLIT,
    seed: int = config.ML_SEED,
    verbose: bool = True,
) -> dict[str, float]:
    if len(ds) == 0:
        raise SystemExit(f"No data in {config.GESTURE_DATA_PATH}. Record some with: python -m ml.record")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feats = ds.features()
    tr, va, te = stratified_split(ds.labels, seed=seed)
    x_val = torch.tensor(feats[va], device=device)
    y_val = torch.tensor(ds.labels[va], device=device)
    num_classes = len(config.GESTURE_CLASSES)

    # Class weights compensate for gestures with fewer samples.
    counts = np.bincount(ds.labels[tr], minlength=num_classes).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / num_classes, 0.0)

    model = build_model(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.ML_LR, weight_decay=config.ML_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    best_acc, best_state, stale = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(tr)
        for start in range(0, len(order), config.ML_BATCH):
            idx = order[start:start + config.ML_BATCH]
            xb = torch.tensor(augment(feats[idx], rng), device=device)
            yb = torch.tensor(ds.labels[idx], device=device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        val_acc = accuracy(model, x_val, y_val) if len(va) else 0.0
        if verbose:
            print(f"epoch {epoch + 1:3d}  loss {loss.item():.4f}  val acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, stale = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.ML_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.cpu()
    test_acc = accuracy(model, torch.tensor(feats[te]), torch.tensor(ds.labels[te]))

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": list(config.GESTURE_CLASSES),
        "hidden": list(config.ML_HIDDEN),
        "val_acc": best_acc,
        "test_acc": test_acc,
    }, out_pt)
    export_numpy(model, out_npz)
    np.savez(out_test, features=feats[te], labels=ds.labels[te])

    result = {"val_acc": best_acc, "test_acc": test_acc, "train": len(tr), "val": len(va), "test": len(te)}
    if verbose:
        print(f"\nbest val acc {best_acc:.4f}, test acc {test_acc:.4f} "
              f"(train {len(tr)}, val {len(va)}, test {len(te)}) on {device}")
        print(f"saved {out_pt}, {out_npz}; run `python -m ml.evaluate` for details")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=config.GESTURE_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=config.ML_EPOCHS)
    args = parser.parse_args()
    ds = load(args.data)
    print("samples per gesture:", ds.counts())
    train(ds, epochs=args.epochs)


if __name__ == "__main__":
    main()
