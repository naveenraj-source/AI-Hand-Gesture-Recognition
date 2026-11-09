"""Rule-based hand gesture classifier.

Works on the 21 hand landmarks produced by MediaPipe Hands, but has no
dependency on MediaPipe or OpenCV, so it can be unit-tested on its own.

Landmarks are a sequence of 21 ``(x, y)`` points (any consistent scale, e.g.
pixels). Index layout follows MediaPipe:

    0 wrist
    1-4   thumb  (CMC, MCP, IP, TIP)
    5-8   index  (MCP, PIP, DIP, TIP)
    9-12  middle (MCP, PIP, DIP, TIP)
    13-16 ring   (MCP, PIP, DIP, TIP)
    17-20 pinky  (MCP, PIP, DIP, TIP)

Finger state is decided from distances, not raw y-coordinates, so the four
finger checks work for any hand rotation and for either hand.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Deque, Dict, List, Sequence, Tuple

Point = Tuple[float, float]

OPEN_HAND = "Open Hand"
FIST = "Fist"
THUMBS_UP = "Thumbs Up"
VICTORY = "Victory / Peace"
UNKNOWN = "Unknown"

NUM_LANDMARKS = 21

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# (PIP, TIP) landmark indices for index, middle, ring, pinky.
FINGERS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}

# A finger counts as extended when its tip is this much farther from the
# wrist than its PIP joint.
FINGER_EXTEND_RATIO = 1.1
# Thumb counts as extended when its tip is farther from the pinky base than
# its IP joint is.
THUMB_EXTEND_RATIO = 1.05


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(landmarks: Sequence[Point]) -> Dict[str, bool]:
    """Return ``{finger_name: is_extended}`` for all five fingers."""
    if len(landmarks) != NUM_LANDMARKS:
        raise ValueError(
            f"Expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}"
        )

    wrist = landmarks[WRIST]
    states: Dict[str, bool] = {}
    for name, (pip, tip) in FINGERS.items():
        states[name] = (
            _dist(landmarks[tip], wrist)
            > _dist(landmarks[pip], wrist) * FINGER_EXTEND_RATIO
        )

    pinky_base = landmarks[PINKY_MCP]
    states["thumb"] = (
        _dist(landmarks[THUMB_TIP], pinky_base)
        > _dist(landmarks[THUMB_IP], pinky_base) * THUMB_EXTEND_RATIO
    )
    return states


def _thumb_points_up(landmarks: Sequence[Point]) -> bool:
    """True if the thumb (MCP -> tip) points mostly upward in the image."""
    mcp = landmarks[THUMB_MCP]
    tip = landmarks[THUMB_TIP]
    dx = tip[0] - mcp[0]
    dy = tip[1] - mcp[1]  # image y grows downward, so "up" means dy < 0
    hand_size = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if hand_size == 0:
        return False
    long_enough = math.hypot(dx, dy) > 0.5 * hand_size
    mostly_vertical = -dy >= abs(dx)
    above_knuckles = tip[1] < landmarks[INDEX_MCP][1]
    return dy < 0 and long_enough and mostly_vertical and above_knuckles


def classify_gesture(landmarks: Sequence[Point]) -> str:
    """Classify a single hand into one of the supported gestures."""
    s = finger_states(landmarks)
    four_extended = s["index"] and s["middle"] and s["ring"] and s["pinky"]
    four_curled = not (s["index"] or s["middle"] or s["ring"] or s["pinky"])

    if four_extended:
        return OPEN_HAND

    if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
        return VICTORY

    if four_curled:
        if s["thumb"]:
            return THUMBS_UP if _thumb_points_up(landmarks) else UNKNOWN
        return FIST

    return UNKNOWN


class GestureSmoother:
    """Majority vote over the last ``size`` predictions to stop flicker."""

    def __init__(self, size: int = 7) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        self._history: Deque[str] = deque(maxlen=size)

    def update(self, label: str) -> str:
        self._history.append(label)
        return Counter(self._history).most_common(1)[0][0]

    def reset(self) -> None:
        self._history.clear()


def landmarks_to_points(
    landmark_list, width: int, height: int
) -> List[Point]:
    """Convert MediaPipe normalized landmarks to pixel-space points.

    Pixel space keeps distances undistorted for non-square frames.
    """
    return [(lm.x * width, lm.y * height) for lm in landmark_list]
