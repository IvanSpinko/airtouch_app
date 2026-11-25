import numpy as np
import pytest
from handgen import FIST_THUMB, POSE_KWARGS, make_hand, pose

from gesture_control import config
from gesture_control.gestures import (
    CLICK_POSES, Gesture, GestureDetector, HandFeatures, classify_pose, compute_features,
    normalize_landmarks,
)

STABLE = max(config.POSE_STABLE_FRAMES, config.SCROLL_STABLE_FRAMES, config.CLICK_POSE_FRAMES)


def feed(detector, hand, frames, handedness="Right"):
    g = None
    for _ in range(frames):
        g = detector.update(hand, handedness)
    return g


def ready_detector(classifier=None):
    """Detector that has already been tracking a hand (past the acquisition grace)."""
    d = GestureDetector(classifier)
    feed(d, pose("l_pose"), config.ACQUIRE_GRACE_FRAMES + 1)
    return d


# -- rules ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", POSE_KWARGS)
def test_rules_recognize_every_gesture(name):
    assert classify_pose(compute_features(pose(name))) == Gesture(name)
    assert feed(ready_detector(), pose(name), STABLE) == Gesture(name)


@pytest.mark.parametrize("name", POSE_KWARGS)
def test_left_hand_gives_same_gesture(name):
    hand = pose(name)
    hand[:, 0] = 1.0 - hand[:, 0]
    assert classify_pose(compute_features(hand)) == Gesture(name)


@pytest.mark.parametrize("scale", [0.6, 1.0, 1.5])
@pytest.mark.parametrize("offset", [(-0.15, -0.1), (0.0, 0.0), (0.1, 0.05)])
def test_features_invariant_to_distance_and_position(scale, offset):
    ref = compute_features(pose("thumb_bent"))
    f = compute_features(pose("thumb_bent", scale=scale, offset=offset))
    assert f.thumb_reach == pytest.approx(ref.thumb_reach, rel=1e-4)
    assert f.extended == ref.extended


def test_thumb_reach_separates_l_pose_from_thumb_bent():
    out = compute_features(pose("l_pose")).thumb_reach
    bent = compute_features(pose("thumb_bent")).thumb_reach
    assert bent < config.THUMB_EXTENDED_RATIO < out


def test_index_bent_is_not_a_fist():
    """Both have the four fingers folded; the straight thumb tells them apart."""
    f = compute_features(pose("index_bent"))
    assert f.is_fist and f.thumb_out
    assert classify_pose(f) == Gesture.INDEX_BENT


def test_finger_pointed_at_camera_is_not_a_fist():
    """Foreshortened in 2D, but depth shows the finger is extended."""
    hand = make_hand((), thumb_tip=FIST_THUMB)
    for k, depth in zip((6, 7, 8), (-0.07, -0.12, -0.16)):
        hand[k, :2] = hand[5, :2]
        hand[k, 2] = depth / config.FRAME_ASPECT
    f = compute_features(hand)
    assert f.extended["index"] and not f.is_fist


def test_fist_requires_fingertips_near_palm():
    folded = {name: False for name in config.FINGER_JOINTS}
    near = HandFeatures(1.0, 1.0, folded, max_tip_spread=config.FIST_TIP_RATIO * 0.5)
    far = HandFeatures(1.0, 1.0, folded, max_tip_spread=config.FIST_TIP_RATIO * 2)
    assert near.is_fist and not far.is_fist
    assert classify_pose(far) != Gesture.FIST


def test_spread_fingers_are_not_scroll():
    hand = pose("two_fingers")
    hand[12, 0] += 0.15 / config.FRAME_ASPECT  # V-sign: middle tip far from index tip
    assert compute_features(hand).index_middle_gap > config.SCROLL_FINGERS_TOGETHER_RATIO
    assert classify_pose(compute_features(hand)) != Gesture.TWO_FINGERS


# -- debounce / timing -----------------------------------------------------------------
def test_click_pose_needs_confirmation_frames():
    d = ready_detector()
    seen = [d.update(pose("thumb_bent")) for _ in range(config.CLICK_POSE_FRAMES)]
    assert Gesture.THUMB_BENT not in seen[:-1] and seen[-1] == Gesture.THUMB_BENT


def test_single_glitch_frame_changes_nothing():
    d = ready_detector()
    feed(d, pose("l_pose"), 5)
    d.update(pose("thumb_bent"))
    assert feed(d, pose("l_pose"), 5) == Gesture.L_POSE


def test_scroll_needs_longer_confirmation():
    d = ready_detector()
    seen = [d.update(pose("two_fingers")) for _ in range(config.SCROLL_STABLE_FRAMES)]
    assert Gesture.TWO_FINGERS not in seen[:-1] and seen[-1] == Gesture.TWO_FINGERS


def test_no_click_pose_right_after_hand_appears():
    d = GestureDetector()
    seen = [d.update(pose("thumb_bent")) for _ in range(config.ACQUIRE_GRACE_FRAMES)]
    assert Gesture.THUMB_BENT not in seen


def test_short_hand_loss_keeps_gesture():
    d = ready_detector()
    feed(d, pose("thumb_bent"), STABLE)
    for _ in range(config.HAND_LOST_GRACE_FRAMES):
        assert d.update(None) == Gesture.THUMB_BENT


def test_long_hand_loss_resets():
    d = ready_detector()
    feed(d, pose("thumb_bent"), STABLE)
    assert feed(d, None, config.HAND_LOST_GRACE_FRAMES + 1) == Gesture.NONE


def test_strength_without_model_marks_the_click_pose():
    d = ready_detector()
    d.update(pose("thumb_bent"))
    assert d.strength == {Gesture.THUMB_BENT: 1.0, Gesture.INDEX_BENT: 0.0}


# -- with a classifier --------------------------------------------------------------------
class FakeClassifier:
    """Returns fixed probabilities: {gesture: prob}, the rest share what is left."""

    def __init__(self, probs):
        self.calls = []
        self.set(probs)

    def set(self, probs):
        rest = (1.0 - sum(probs.values())) / (len(Gesture) - len(probs))
        self.probs = {g: probs.get(g, rest) for g in Gesture}

    def predict(self, landmarks, handedness="Right"):
        self.calls.append(handedness)
        return dict(self.probs)


def model_detector(probs):
    clf = FakeClassifier({Gesture.L_POSE: 0.95})
    d = ready_detector(clf)
    clf.set(probs)
    return d, clf


def test_confident_model_decides():
    d, _ = model_detector({Gesture.SHAKA: 0.95})
    assert feed(d, pose("l_pose"), STABLE + 3) == Gesture.SHAKA  # rules say L, model wins


def test_unsure_model_falls_back_to_rules():
    d, _ = model_detector({Gesture.SHAKA: 0.3, Gesture.FIST: 0.3})
    assert feed(d, pose("two_fingers"), STABLE + 3) == Gesture.TWO_FINGERS


def test_smoothing_ignores_a_single_odd_frame():
    d, clf = model_detector({Gesture.L_POSE: 0.95})
    feed(d, pose("l_pose"), 5)
    clf.set({Gesture.THUMB_BENT: 0.95})
    d.update(pose("l_pose"))  # one odd model frame
    clf.set({Gesture.L_POSE: 0.95})
    assert feed(d, pose("l_pose"), 3) == Gesture.L_POSE
    assert d.strength[Gesture.THUMB_BENT] < config.CLASSIFIER_MIN_PROB


def test_smoothing_follows_a_held_change():
    d, clf = model_detector({Gesture.L_POSE: 0.95})
    feed(d, pose("l_pose"), 5)
    clf.set({Gesture.THUMB_BENT: 0.95})
    assert feed(d, pose("l_pose"), 6) == Gesture.THUMB_BENT


def test_detector_passes_handedness_to_model():
    d, clf = model_detector({Gesture.L_POSE: 0.95})
    d.update(pose("l_pose"), "Left")
    assert clf.calls[-1] == "Left"


def test_strength_comes_from_model():
    d, _ = model_detector({Gesture.INDEX_BENT: 0.8})
    feed(d, pose("l_pose"), 10)
    assert d.strength[Gesture.INDEX_BENT] == pytest.approx(0.8, abs=0.01)
    assert set(d.strength) == set(CLICK_POSES)


# -- normalization for the classifier ------------------------------------------------------
def rotate(hand, deg):
    a = np.deg2rad(deg)
    out = hand.copy()
    cx, cy = 0.5 * config.FRAME_ASPECT, 0.5
    x = hand[:, 0] * config.FRAME_ASPECT - cx
    y = hand[:, 1] - cy
    out[:, 0] = (np.cos(a) * x - np.sin(a) * y + cx) / config.FRAME_ASPECT
    out[:, 1] = np.sin(a) * x + np.cos(a) * y + cy
    return out


def test_normalize_is_invariant_to_position_size_rotation():
    ref = normalize_landmarks(pose("two_fingers"))
    for hand in (
        pose("two_fingers", offset=(0.1, -0.1)),
        pose("two_fingers", scale=0.5),
        rotate(pose("two_fingers"), 30),
    ):
        assert np.allclose(normalize_landmarks(hand), ref, atol=1e-4)


def test_normalize_maps_left_hand_onto_right():
    right = pose("thumb_bent")
    left = right.copy()
    left[:, 0] = 1.0 - left[:, 0]
    assert np.allclose(
        normalize_landmarks(left, "Left"), normalize_landmarks(right, "Right"), atol=1e-5
    )


def test_normalize_shape_and_anchor():
    f = normalize_landmarks(pose("l_pose")).reshape(-1, 3)
    assert f.shape == (config.NUM_LANDMARKS, 3)
    assert np.allclose(f[config.LM_WRIST], 0.0)
    assert np.allclose(f[config.LM_MIDDLE_MCP, :2], (0.0, -1.0), atol=1e-5)
