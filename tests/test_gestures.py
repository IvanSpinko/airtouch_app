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


def ready_detector():
    """Detector that has already been tracking a hand (past the acquisition grace)."""
    d = GestureDetector()
    feed(d, make_hand(("index",)), config.ACQUIRE_GRACE_FRAMES + 1)
    return d


def feed(detector, hand, frames):
    g = None
    for _ in range(frames):
        g = detector.update(hand)
    return g


STABLE = max(config.POSE_STABLE_FRAMES, config.SCROLL_STABLE_FRAMES)


def click_ready_hand():
    """User's pose before a left click: ring + pinky folded, index pointing, middle
    spread out towards the thumb which is ready to pinch it."""
    hand = make_hand(("index", "middle"))
    for k, (x, y) in zip((10, 11, 12), ((0.53, 0.64), (0.55, 0.60), (0.57, 0.57))):
        hand[k, :2] = (x / config.FRAME_ASPECT, y)
    hand[4, :2] = (0.45 / config.FRAME_ASPECT, 0.66)
    return hand


@pytest.mark.parametrize("name", POSES)
def test_each_pose_is_recognized(name):
    hand, expected = POSES[name]
    assert feed(ready_detector(), hand, STABLE) == expected


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
        assert feed(ready_detector(), mirrored, STABLE) == expected


def test_pinch_needs_confirmation_frames():
    d = ready_detector()
    feed(d, POSES["point"][0], config.POSE_STABLE_FRAMES)
    pinch = POSES["pinch_index"][0]
    results = [d.update(pinch) for _ in range(config.PINCH_CONFIRM_FRAMES)]
    assert results[-1] == Gesture.PINCH_INDEX
    assert Gesture.PINCH_INDEX not in results[:-1]


def test_single_glitch_frame_is_not_a_click():
    d = ready_detector()
    point, pinch = POSES["point"][0], POSES["pinch_index"][0]
    feed(d, point, 5)
    d.update(pinch)
    assert feed(d, point, 5) == Gesture.POINT


def test_pinch_hysteresis_holds_between_thresholds():
    d = ready_detector()
    feed(d, POSES["pinch_index"][0], config.PINCH_CONFIRM_FRAMES)
    # open the pinch to a distance between ON and OFF: still pinched
    mid = (config.PINCH_ON_RATIO + config.PINCH_OFF_RATIO) / 2
    hand = make_hand(("index",))
    palm = 0.2
    tip = np.array([hand[8, 0] * config.FRAME_ASPECT, hand[8, 1]])
    hand_mid = make_hand(("index",), thumb_tip=tuple(tip + (mid * palm, 0.0)))
    assert compute_features(hand_mid).pinch_index == pytest.approx(mid, rel=1e-3)
    assert d.update(hand_mid) == Gesture.PINCH_INDEX
    # fully open: released after the debounce
    assert feed(d, hand, config.PINCH_RELEASE_FRAMES) != Gesture.PINCH_INDEX


def test_fist_never_triggers_pinch():
    d = ready_detector()
    fist = POSES["fist"][0]
    assert compute_features(fist).pinch_index < config.PINCH_ON_RATIO  # thumb is near
    results = [d.update(fist) for _ in range(10)]
    assert Gesture.PINCH_INDEX not in results and Gesture.PINCH_MIDDLE not in results
    assert results[-1] == Gesture.FIST


def test_short_hand_loss_keeps_gesture():
    d = ready_detector()
    feed(d, POSES["pinch_index"][0], 3)
    for _ in range(config.HAND_LOST_GRACE_FRAMES):
        assert d.update(None) == Gesture.PINCH_INDEX
    assert d.update(POSES["pinch_index"][0]) == Gesture.PINCH_INDEX


def test_long_hand_loss_resets():
    d = ready_detector()
    feed(d, POSES["pinch_index"][0], 3)
    assert feed(d, None, config.HAND_LOST_GRACE_FRAMES + 1) == Gesture.NONE
    assert d.update(POSES["pinch_index"][0]) != Gesture.PINCH_INDEX  # needs confirmation again


def test_classify_pose_is_stateless():
    f = compute_features(POSES["two"][0])
    assert classify_pose(f) == classify_pose(f) == Gesture.TWO_FINGERS


def test_finger_pointed_at_camera_is_not_a_fist():
    """Foreshortened in 2D, but depth shows the finger is extended."""
    hand = make_hand((), thumb_tip=FIST_THUMB)
    # index finger sticks out towards the camera: tip/dip/pip in front of the palm
    for k, depth in zip((6, 7, 8), (-0.07, -0.12, -0.16)):
        hand[k, :2] = hand[5, :2]
        hand[k, 2] = depth / config.FRAME_ASPECT
    f = compute_features(hand)
    assert f.extended["index"]
    assert not f.is_fist
    assert feed(ready_detector(), hand, config.POSE_STABLE_FRAMES) != Gesture.FIST


def test_fist_requires_fingertips_near_palm():
    from gesture_control.gestures import HandFeatures

    folded = {name: False for name in config.FINGER_JOINTS}
    near = HandFeatures(1.0, 1.0, folded, max_tip_spread=config.FIST_TIP_RATIO * 0.5)
    far = HandFeatures(1.0, 1.0, folded, max_tip_spread=config.FIST_TIP_RATIO * 2)
    assert near.is_fist
    assert not far.is_fist
    assert classify_pose(far) != Gesture.FIST


def test_click_ready_pose_is_not_scroll():
    hand = click_ready_hand()
    f = compute_features(hand)
    assert f.extended["index"] and f.extended["middle"]
    assert not f.extended["ring"] and not f.extended["pinky"]
    assert f.pinch_middle > config.PINCH_ON_RATIO  # not pinched yet
    assert feed(ready_detector(), hand, 20) == Gesture.POINT


def test_click_ready_then_pinch_never_scrolls():
    d = ready_detector()
    seen = [d.update(click_ready_hand()) for _ in range(10)]
    seen += [d.update(POSES["pinch_middle"][0]) for _ in range(5)]
    assert Gesture.TWO_FINGERS not in seen
    assert seen[-1] == Gesture.PINCH_MIDDLE


def test_spread_fingers_are_not_scroll():
    hand = make_hand(("index", "middle"))
    hand[12, 0] += 0.15 / config.FRAME_ASPECT  # V-sign: middle tip far from index tip
    assert compute_features(hand).index_middle_gap > config.SCROLL_FINGERS_TOGETHER_RATIO
    assert feed(ready_detector(), hand, 20) != Gesture.TWO_FINGERS


def test_scroll_needs_longer_confirmation_than_other_poses():
    d = ready_detector()
    two = POSES["two"][0]
    seen = [d.update(two) for _ in range(config.SCROLL_STABLE_FRAMES)]
    assert Gesture.TWO_FINGERS not in seen[:-1]
    assert seen[-1] == Gesture.TWO_FINGERS



def test_no_click_right_after_hand_appears():
    d = GestureDetector()
    pinch = POSES["pinch_middle"][0]
    seen = [d.update(pinch) for _ in range(config.ACQUIRE_GRACE_FRAMES)]
    assert Gesture.PINCH_MIDDLE not in seen


def test_pinch_flicker_does_not_release():
    d = ready_detector()
    pinch, open_ = POSES["pinch_middle"][0], POSES["point"][0]
    feed(d, pinch, config.PINCH_CONFIRM_FRAMES)
    for _ in range(5):  # one open frame every other frame
        assert d.update(open_) == Gesture.PINCH_MIDDLE
        assert d.update(pinch) == Gesture.PINCH_MIDDLE


def test_pinch_releases_after_debounce():
    d = ready_detector()
    feed(d, POSES["pinch_middle"][0], config.PINCH_CONFIRM_FRAMES)
    results = [d.update(POSES["point"][0]) for _ in range(config.PINCH_RELEASE_FRAMES)]
    assert results[-1] != Gesture.PINCH_MIDDLE
