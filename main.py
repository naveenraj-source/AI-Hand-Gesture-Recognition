"""Real-time hand gesture recognition from a webcam.

Usage:
    python main.py                 # webcam 0
    python main.py --camera 1      # another camera
    python main.py --image hand.jpg --output result.jpg   # single image

Keys (webcam mode):  q / ESC = quit,  s = save screenshot
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp

from gesture_classifier import (
    UNKNOWN,
    GestureSmoother,
    classify_gesture,
    landmarks_to_points,
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

WINDOW_NAME = "AI Hand Gesture Recognition"
SCREENSHOT_DIR = "screenshots"

LABEL_COLORS = {
    "Open Hand": (80, 200, 80),
    "Fist": (60, 60, 230),
    "Thumbs Up": (230, 160, 40),
    "Victory / Peace": (200, 80, 200),
    UNKNOWN: (150, 150, 150),
}


def draw_label(frame, text: str, fps: float | None = None) -> None:
    color = LABEL_COLORS.get(text, LABEL_COLORS[UNKNOWN])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (30, 30, 30), -1)
    cv2.putText(
        frame, f"Gesture: {text}", (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA,
    )
    if fps is not None:
        cv2.putText(
            frame, f"{fps:4.1f} FPS", (frame.shape[1] - 130, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
    cv2.putText(
        frame, "q: quit   s: screenshot", (15, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )


def process_frame(frame, hands, smoother: GestureSmoother | None):
    """Detect a hand in a BGR frame, draw it, return the gesture label."""
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        if smoother is not None:
            smoother.reset()
        return "No hand detected"

    hand = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(
        frame,
        hand,
        mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )
    points = landmarks_to_points(hand.landmark, width, height)
    label = classify_gesture(points)
    return smoother.update(label) if smoother is not None else label


def run_image(path: str, output: str) -> int:
    frame = cv2.imread(path)
    if frame is None:
        print(f"Error: could not read image '{path}'.", file=sys.stderr)
        return 1
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        label = process_frame(frame, hands, None)
    draw_label(frame, label)
    cv2.imwrite(output, frame)
    print(f"Detected: {label}  ->  saved to {output}")
    return 0


def run_webcam(camera_index: int, flip: bool) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(
            f"Error: cannot open camera {camera_index}. Check that a webcam "
            "is connected, not used by another app, and that camera access "
            "is allowed. Try --camera 1 for an external webcam.",
            file=sys.stderr,
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother = GestureSmoother(size=7)
    prev_time = time.time()
    fps = 0.0
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read a frame from the camera.",
                      file=sys.stderr)
                break

            if flip:
                frame = cv2.flip(frame, 1)  # mirror view feels natural

            label = process_frame(frame, hands, smoother)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            draw_label(frame, label, fps)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
                out = os.path.join(SCREENSHOT_DIR, name)
                cv2.imwrite(out, frame)
                print(f"Saved {out}")

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Hand Gesture Recognition")
    parser.add_argument("--camera", type=int, default=0,
                        help="webcam index (default 0)")
    parser.add_argument("--no-flip", action="store_true",
                        help="do not mirror the webcam image")
    parser.add_argument("--image", type=str,
                        help="classify a single image instead of the webcam")
    parser.add_argument("--output", type=str, default="result.jpg",
                        help="output path for --image mode")
    args = parser.parse_args()

    if args.image:
        return run_image(args.image, args.output)
    return run_webcam(args.camera, flip=not args.no_flip)


if __name__ == "__main__":
    sys.exit(main())
