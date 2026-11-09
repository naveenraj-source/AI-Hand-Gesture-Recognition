# AI Hand Gesture Recognition

Real-time hand gesture recognition using a webcam. The app detects one hand,
classifies the gesture and shows it on screen.

**Recognized gestures:** Open Hand, Fist, Thumbs Up, Victory / Peace
(anything else is shown as "Unknown", and "No hand detected" when no hand is visible).

## Technology used

| Part | Tool | Role |
|------|------|------|
| Video capture and display | OpenCV | Reads webcam frames, draws the overlay, shows the window |
| Hand detection and tracking | Google MediaPipe Hands | A pre-trained ML model that finds the hand and returns 21 landmarks (finger joints and tips) per frame |
| Gesture classification | Python (`gesture_classifier.py`) | Rule-based logic on the landmarks |
| Stabilisation | Majority vote over the last 7 frames | Stops the label from flickering |

### How the classification works

1. MediaPipe gives 21 `(x, y)` landmarks, converted to pixel coordinates.
2. A **finger is "extended"** when its tip is farther from the wrist than its middle joint (PIP). Using distances instead of raw heights makes this work at any hand rotation and for either hand.
3. The **thumb is "extended"** when its tip is farther from the pinky base than its IP joint.
4. Rules:
   - four fingers extended -> **Open Hand**
   - index + middle extended, ring + pinky curled -> **Victory / Peace**
   - four fingers curled, thumb extended and pointing up -> **Thumbs Up**
   - four fingers curled, thumb not extended -> **Fist**
   - everything else -> **Unknown**

## Project structure

```
hand-gesture-recognition/
├── main.py                  # webcam app / image mode
├── gesture_classifier.py    # classification logic (no OpenCV/MediaPipe needed)
├── tests/test_gesture_classifier.py
├── requirements.txt
├── screenshots/             # put your screenshots here
├── LICENSE
└── README.md
```

## Setup

Requires Python 3.9 - 3.12 and a webcam.

```bash
git clone https://github.com/<your-username>/hand-gesture-recognition.git
cd hand-gesture-recognition

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py                # default webcam
python main.py --camera 1     # external webcam
python main.py --no-flip      # disable mirror view
python main.py --image hand.jpg --output result.jpg   # classify one photo
```

Keys in the webcam window: **q** or **Esc** = quit, **s** = save a screenshot to `screenshots/`.

## Run the tests

The tests use synthetic landmarks, so no webcam is needed:

```bash
python -m unittest discover -s tests -v
```

## Tips for best results

- Good, even lighting; plain background helps.
- Keep the hand fully in view, about 40 - 100 cm from the camera.
- Face the palm toward the camera for Open Hand and Victory. For Thumbs Up, keep the thumb pointing upward.

## Troubleshooting

- **"cannot open camera"**: close other apps using the webcam, allow camera permission for your terminal/IDE, try `--camera 1`.
- **`module 'mediapipe' has no attribute 'solutions'`** or **no matching mediapipe version**: you are on Python 3.13+ or a newer MediaPipe. Use Python 3.9 - 3.12 and `pip install -r requirements.txt` (pinned to `mediapipe==0.10.21`).
- **numpy errors on import**: `pip install "numpy<2"`.

## Screenshots

Add your own screenshots to `screenshots/` (press **s** while running) and link them here:

```
![Open Hand](screenshots/open_hand.png)
![Fist](screenshots/fist.png)
![Thumbs Up](screenshots/thumbs_up.png)
![Victory](screenshots/victory.png)
```

## Limitations

- Rule-based, so unusual angles (e.g. hand pointing at the camera) can be misclassified.
- Tracks one hand at a time.

## License

MIT
