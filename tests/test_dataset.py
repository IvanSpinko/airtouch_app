import numpy as np
from handgen import make_hand

from gesture_control import config
from ml.dataset import append_rows, augment, load, remove_last_rows, stratified_split


def write(path, n_per_class=20):
    rows = []
    for name in config.GESTURE_CLASSES:
        for _ in range(n_per_class):
            rows.append((name, "Right", "s1", make_hand(("index",))))
    append_rows(path, rows)


def test_roundtrip(tmp_path):
    path = tmp_path / "g.csv"
    lm = make_hand(("index", "middle"))
    append_rows(path, [("l_pose", "Left", "s1", lm)])
    ds = load(path)
    assert len(ds) == 1
    assert ds.labels[0] == config.GESTURE_CLASSES.index("l_pose")
    assert ds.handedness[0] == "Left"
    assert np.allclose(ds.landmarks[0], lm, atol=1e-5)
    assert ds.features().shape == (1, config.NUM_LANDMARKS * 3)


def test_append_keeps_single_header(tmp_path):
    path = tmp_path / "g.csv"
    write(path, 2)
    write(path, 2)
    assert len(load(path)) == 2 * 2 * len(config.GESTURE_CLASSES)


def test_counts(tmp_path):
    path = tmp_path / "g.csv"
    write(path, 3)
    assert set(load(path).counts().values()) == {3}


def test_undo_removes_last_rows_only(tmp_path):
    path = tmp_path / "g.csv"
    write(path, 2)
    before = len(load(path))
    assert remove_last_rows(path, 5) == 5
    assert len(load(path)) == before - 5
    assert remove_last_rows(path, 10_000) == before - 5  # never the header
    assert len(load(path)) == 0
    assert remove_last_rows(tmp_path / "missing.csv", 3) == 0


def test_missing_file_is_empty_dataset(tmp_path):
    ds = load(tmp_path / "nope.csv")
    assert len(ds) == 0 and ds.features().shape == (0, config.NUM_LANDMARKS * 3)


def test_stratified_split_is_disjoint_and_balanced():
    labels = np.repeat(np.arange(4), 100)
    tr, va, te = stratified_split(labels, val=0.15, test=0.15, seed=1)
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == len(labels)
    for part, frac in ((va, 0.15), (te, 0.15)):
        assert all(abs((labels[part] == c).sum() - 100 * frac) <= 1 for c in range(4))


def test_stratified_split_is_deterministic():
    labels = np.repeat(np.arange(3), 50)
    a = stratified_split(labels, seed=7)
    b = stratified_split(labels, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_augment_keeps_shape_and_changes_values():
    x = np.tile(np.arange(63, dtype=np.float32), (10, 1)) / 63
    y = augment(x, np.random.default_rng(0))
    assert y.shape == x.shape and y.dtype == np.float32
    assert not np.allclose(x, y)
