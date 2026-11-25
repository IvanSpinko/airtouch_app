"""Synthetic hand landmarks for camera-free tests.

Builds a schematic upright hand (fingers pointing up in the image) in the same
format the tracker produces: (21, 3) array, x normalized by frame width,
y by frame height.
"""

from __future__ import annotations

import numpy as np

from gesture_control import config

WRIST = (0.50, 0.90)
THUMB_BASE = [(0.42, 0.85), (0.37, 0.80), (0.34, 0.75)]  # CMC, MCP, IP
THUMB_TIP_AWAY = (0.31, 0.70)
MCP = {"index": (0.44, 0.71), "middle": (0.50, 0.70), "ring": (0.56, 0.71), "pinky": (0.61, 0.73)}
FIRST_LM = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
EXTENDED = [(0, -0.07), (0, -0.12), (0, -0.16)]  # PIP, DIP, TIP offsets from MCP
FOLDED = [(0, -0.05), (0, -0.02), (0, 0.04)]


def make_hand(
    extended=("index",),
    thumb_tip=None,
    offset=(0.0, 0.0),
    scale=1.0,
    aspect=config.FRAME_ASPECT,
) -> np.ndarray:
    """Hand with the given fingers extended; thumb tip far away unless given.

    ``thumb_tip`` may be a point or the name of a finger whose tip it should touch.
    Coordinates are built in square units, then x is divided by ``aspect``.
    """
    pts = np.zeros((config.NUM_LANDMARKS, 2))
    pts[0] = WRIST
    pts[1:4] = THUMB_BASE
    for name, mcp in MCP.items():
        i = FIRST_LM[name]
        pts[i] = mcp
        for k, (dx, dy) in enumerate(EXTENDED if name in extended else FOLDED, start=1):
            pts[i + k] = (mcp[0] + dx, mcp[1] + dy)
    if thumb_tip is None:
        pts[4] = THUMB_TIP_AWAY
    elif isinstance(thumb_tip, str):
        pts[4] = pts[FIRST_LM[thumb_tip] + 3] + (0.01, 0.01)
    else:
        pts[4] = thumb_tip

    center = np.array(WRIST)
    pts = (pts - center) * scale + center + np.array(offset)
    pts[:, 0] /= aspect
    out = np.zeros((config.NUM_LANDMARKS, 3), dtype=np.float32)
    out[:, :2] = pts
    return out


FIST_THUMB = (0.47, 0.76)
ALL = ("index", "middle", "ring", "pinky")
# Thumb tip bent in across the palm (left-click pose), square units.
THUMB_IN = (0.50, 0.77)

# One synthetic hand per gesture class of the current gesture set.
POSE_KWARGS = {
    "none": dict(extended=("ring", "pinky")),
    "l_pose": dict(extended=("index",)),
    "thumb_bent": dict(extended=("index",), thumb_tip=THUMB_IN),
    "index_bent": dict(extended=()),
    "two_fingers": dict(extended=("index", "middle")),
    "fist": dict(extended=(), thumb_tip=FIST_THUMB),
    "open_palm": dict(extended=ALL),
    "shaka": dict(extended=("pinky",)),
}


def pose(name, **kw):
    return make_hand(**{**POSE_KWARGS[name], **kw})
