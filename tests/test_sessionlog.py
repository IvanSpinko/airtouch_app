"""Session log round trip and replay (no camera, no mouse)."""

from handgen import make_hand, pose

from gesture_control.sessionlog import SessionLogger
from ml.replay import load_log, replay

DT = 1 / 30


def test_log_roundtrip_and_replay(tmp_path):
    log = SessionLogger(tmp_path)
    lpose, bent = pose("l_pose"), pose("thumb_bent")
    frames = [lpose] * 15 + [bent] * 4 + [lpose] * 10 + [None] * 3
    for i, lm in enumerate(frames):
        log.write(i * DT, lm, "Right", "l_pose", 0.0, 0.0, None)
    log.mark(len(frames) * DT)
    log.close()

    loaded, marks = load_log(log.path)
    assert len(loaded) == len(frames) and len(marks) == 1
    assert loaded[-1].landmarks is None and loaded[0].landmarks.shape == (21, 3)

    trace = replay(loaded, classifier=None)
    events = [r["event"] for r in trace if r["event"]]
    assert events == ["LEFT CLICK"]  # thumb bend + return to the L pose


def test_logger_writes_each_event_once(tmp_path):
    log = SessionLogger(tmp_path)
    lm = make_hand(("index",))
    ev = ("LEFT CLICK", 1.0)
    for i in range(3):
        log.write(i * DT, lm, "Right", "l_pose", 0.0, 0.0, ev)
    log.close()
    loaded, _ = load_log(log.path)
    assert [f.logged_event for f in loaded] == ["LEFT CLICK", "", ""]
