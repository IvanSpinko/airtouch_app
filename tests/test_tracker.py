import numpy as np
from handgen import make_hand

from gesture_control import config
from gesture_control.tracker import Hand, near_edges


def hand_at(offset=(0.0, 0.0)):
    return Hand(make_hand(("index",), offset=offset), "Right", 1.0)


def test_centered_hand_is_not_near_any_edge():
    h = hand_at((0.0, -0.2))
    assert near_edges(h) == set()


def test_detects_each_edge():
    assert "left" in near_edges(hand_at((-0.45, -0.2)))
    assert "right" in near_edges(hand_at((0.7, -0.2)))  # offsets are in square units
    assert "top" in near_edges(hand_at((0.0, -0.55)))
    assert "bottom" in near_edges(hand_at((0.0, 0.07)))


def test_margin_is_respected():
    lm = np.full((config.NUM_LANDMARKS, 3), 0.5, dtype=np.float32)
    lm[0, 0] = 0.03
    h = Hand(lm, "Left", 1.0)
    assert near_edges(h, margin=0.05) == {"left"}
    assert near_edges(h, margin=0.02) == set()
