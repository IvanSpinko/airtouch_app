"""End-to-end smoke test of the ML pipeline on synthetic hands (no camera)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from handgen import POSE_KWARGS, make_hand  # noqa: E402

from gesture_control import config  # noqa: E402
from gesture_control.gestures import GestureClassifier, normalize_landmarks  # noqa: E402
from ml.dataset import append_rows, load  # noqa: E402
from ml.evaluate import evaluate  # noqa: E402
from ml.train import build_model, train  # noqa: E402

POSES = POSE_KWARGS  # one synthetic hand per gesture class


def synthetic_rows(n_per_class, rng):
    rows = []
    for name, kw in POSES.items():
        for _ in range(n_per_class):
            lm = make_hand(
                offset=tuple(rng.uniform(-0.1, 0.1, 2)), scale=rng.uniform(0.7, 1.3), **kw
            )
            lm[:, :2] += rng.normal(0, 0.004, (config.NUM_LANDMARKS, 2))
            hand = "Right"
            if rng.random() < 0.5:
                lm[:, 0] = 1.0 - lm[:, 0]
                hand = "Left"
            rows.append((name, hand, "synthetic", lm))
    return rows


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("ml")
    data = d / "gestures.csv"
    append_rows(data, synthetic_rows(60, np.random.default_rng(0)))
    out = dict(out_pt=d / "g.pt", out_npz=d / "g.npz", out_test=d / "split.npz")
    result = train(load(data), epochs=30, verbose=False, **out)
    return result, out


def test_training_learns_synthetic_gestures(trained):
    result, _ = trained
    assert result["test_acc"] > 0.9


def test_numpy_runtime_matches_torch(trained):
    _, out = trained
    ckpt = torch.load(out["out_pt"])
    model = build_model(len(config.GESTURE_CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x = np.random.default_rng(1).normal(size=(5, config.NUM_LANDMARKS * 3)).astype(np.float32)
    with torch.no_grad():
        expected = torch.softmax(model(torch.tensor(x)), dim=1).numpy()
    got = GestureClassifier(out["out_npz"]).predict_proba(x)
    assert np.allclose(got, expected, atol=1e-5)


def test_classifier_predicts_real_landmarks(trained):
    _, out = trained
    clf = GestureClassifier(out["out_npz"])
    probs = clf.predict(make_hand(**POSES["fist"]))
    assert max(probs, key=probs.get).value == "fist"
    assert sum(probs.values()) == pytest.approx(1.0)


def test_evaluate_reads_saved_split(trained):
    result, out = trained
    acc, cm = evaluate(out["out_npz"], out["out_test"])
    assert acc == pytest.approx(result["test_acc"], abs=1e-6)
    assert cm.sum() == result["test"]


def test_features_used_in_training_match_runtime():
    lm = make_hand(("index",))
    ds_feat = normalize_landmarks(lm, "Right")
    assert ds_feat.shape == (config.NUM_LANDMARKS * 3,)
