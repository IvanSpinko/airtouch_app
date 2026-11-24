import numpy as np
import pytest
from handgen import ALL, FIST_THUMB, make_hand

from gesture_control import config
from gesture_control.gestures import Gesture, GestureDetector, classify_pose, compute_features

POSES = {
    "point": (make_hand(("index",)), Gesture.POINT),
    "two": (make_hand(("index", "middle")), Gesture.TWO_FINGERS),
    "fist": (make_hand((), thumb_tip=FIST_THUMB), Gesture.FIST),
    "palm": (make_hand(ALL), Gesture.OPEN_PALM),
    "pinch_index": (make_hand(("index",), thumb_tip="index"), Gesture.PINCH_INDEX),
    "pinch_middle": (make_hand(("index", "middle"), thumb_tip="middle"), Gesture.PINCH_MIDDLE),
}


def feed(detector, hand, frames):
    g = None
    for _ in range(frames):
        g = detector.update(hand)
    return g


@pytest.mark.parametrize("name", POSES)
def test_each_pose_is_recognized(name):
    hand, expected = POSES[name]
    assert feed(GestureDetector(), hand, config.POSE_STABLE_FRAMES) == expected


@pytest.mark.parametrize("scale", [0.6, 1.0, 1.5])
@pytest.mark.parametrize("offset", [(-0.15, -0.1), (0.0, 0.0), (0.1, 0.05)])
def test_features_invariant_to_distance_and_position(scale, offset):
    ref = compute_features(make_hand(("index",), thumb_tip="index"))
    f = compute_features(make_hand(("index",), thumb_tip="index", scale=scale, offset=offset))
    assert f.pinch_index == pytest.approx(ref.pinch_index, rel=1e-4)
    assert f.extended == ref.extended


def test_mirrored_hand_gives_same_gesture():
    """Left and right hands are mirror images of each other."""
    for hand, expected in POSES.values():
        mirrored = hand.copy()
        mirrored[:, 0] = 1.0 - mirrored[:, 0]
        assert feed(GestureDetector(), mirrored, config.POSE_STABLE_FRAMES) == expected


def test_pinch_needs_confirmation_frames():
    d = GestureDetector()
    feed(d, POSES["point"][0], config.POSE_STABLE_FRAMES)
    pinch = POSES["pinch_index"][0]
    results = [d.update(pinch) for _ in range(config.PINCH_CONFIRM_FRAMES)]
    assert results[-1] == Gesture.PINCH_INDEX
    assert Gesture.PINCH_INDEX not in results[:-1]


def test_single_glitch_frame_is_not_a_click():
    d = GestureDetector()
    point, pinch = POSES["point"][0], POSES["pinch_index"][0]
    feed(d, point, 5)
    d.update(pinch)
    assert feed(d, point, 5) == Gesture.POINT


def test_pinch_hysteresis_holds_between_thresholds():
    d = GestureDetector()
    feed(d, POSES["pinch_index"][0], config.PINCH_CONFIRM_FRAMES)
    # open the pinch to a distance between ON and OFF: still pinched
    mid = (config.PINCH_ON_RATIO + config.PINCH_OFF_RATIO) / 2
    hand = make_hand(("index",))
    palm = 0.2
    tip = np.array([hand[8, 0] * config.FRAME_ASPECT, hand[8, 1]])
    hand_mid = make_hand(("index",), thumb_tip=tuple(tip + (mid * palm, 0.0)))
    assert compute_features(hand_mid).pinch_index == pytest.approx(mid, rel=1e-3)
    assert d.update(hand_mid) == Gesture.PINCH_INDEX
    # fully open: released
    assert d.update(hand) != Gesture.PINCH_INDEX


def test_fist_never_triggers_pinch():
    d = GestureDetector()
    fist = POSES["fist"][0]
    assert compute_features(fist).pinch_index < config.PINCH_ON_RATIO  # thumb is near
    results = [d.update(fist) for _ in range(10)]
    assert Gesture.PINCH_INDEX not in results and Gesture.PINCH_MIDDLE not in results
    assert results[-1] == Gesture.FIST


def test_short_hand_loss_keeps_gesture():
    d = GestureDetector()
    feed(d, POSES["pinch_index"][0], 3)
    for _ in range(config.HAND_LOST_GRACE_FRAMES):
        assert d.update(None) == Gesture.PINCH_INDEX
    assert d.update(POSES["pinch_index"][0]) == Gesture.PINCH_INDEX


def test_long_hand_loss_resets():
    d = GestureDetector()
    feed(d, POSES["pinch_index"][0], 3)
    assert feed(d, None, config.HAND_LOST_GRACE_FRAMES + 1) == Gesture.NONE
    assert d.update(POSES["pinch_index"][0]) != Gesture.PINCH_INDEX  # needs confirmation again


def test_classify_pose_is_stateless():
    f = compute_features(POSES["two"][0])
    assert classify_pose(f) == classify_pose(f) == Gesture.TWO_FINGERS
