"""Session log round trip and replay (no camera, no mouse)."""

from handgen import make_hand

from gesture_control.sessionlog import SessionLogger
from ml.replay import load_log, replay

DT = 1 / 30


def test_log_roundtrip_and_replay(tmp_path):
    log = SessionLogger(tmp_path)
    point = make_hand(("index",))
    pinch = make_hand(("index",), thumb_tip="index")
    frames = [point] * 15 + [pinch] * 4 + [point] * 10 + [None] * 3
    for i, lm in enumerate(frames):
        log.write(i * DT, lm, "Right", "point", 0.0, 0.0, None)
    log.mark(len(frames) * DT)
    log.close()

    loaded, marks = load_log(log.path)
    assert len(loaded) == len(frames) and len(marks) == 1
    assert loaded[-1].landmarks is None and loaded[0].landmarks.shape == (21, 3)

    trace = replay(loaded, classifier=None)
    events = [r["event"] for r in trace if r["event"]]
    assert events == ["LEFT DOWN", "LEFT CLICK"]


def test_logger_writes_each_event_once(tmp_path):
    log = SessionLogger(tmp_path)
    lm = make_hand(("index",))
    ev = ("LEFT CLICK", 1.0)
    for i in range(3):
        log.write(i * DT, lm, "Right", "point", 0.0, 0.0, ev)
    log.close()
    loaded, _ = load_log(log.path)
    assert [f.logged_event for f in loaded] == ["LEFT CLICK", "", ""]
