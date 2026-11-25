from handgen import make_hand

from ml.dataset import append_rows, load
from ml.migrate import migrate


def test_copies_only_unchanged_classes_once(tmp_path):
    old, new = tmp_path / "v1.csv", tmp_path / "v2.csv"
    lm = make_hand(("index",))
    append_rows(old, [(label, "Right", "s", lm) for label in
                      ("none", "point", "pinch_index", "two_fingers", "fist", "open_palm", "fist")])

    assert migrate(old, new, ("two_fingers", "fist", "open_palm")) == {
        "two_fingers": 1, "fist": 2, "open_palm": 1,
    }
    assert sorted(load(new).counts().items()) == sorted({
        "none": 0, "l_pose": 0, "thumb_bent": 0, "index_bent": 0,
        "two_fingers": 1, "fist": 2, "open_palm": 1, "shaka": 0,
    }.items())
    # second run adds nothing
    assert migrate(old, new, ("two_fingers", "fist", "open_palm")) == {
        "two_fingers": 0, "fist": 0, "open_palm": 0,
    }
    assert len(load(new)) == 4


def test_missing_source_is_noop(tmp_path):
    assert migrate(tmp_path / "nope.csv", tmp_path / "v2.csv") == {}
