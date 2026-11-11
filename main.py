"""
AI Hand Gesture Recognition
Flask Web API with Live In-Browser Camera & Desktop Webcam support.

Endpoints:
    GET  /          Interactive Live Camera UI / API information
    GET  /health    Health check
    POST /predict   Accepts image file (form-data) or Base64 (JSON) for gesture recognition
"""

import argparse
import base64
import os
import sys
import tempfile
import webbrowser

# Configure writable cache directory for serverless environments (e.g. Vercel)
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from flask import Flask, jsonify, render_template_string, request
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing required dependencies. Install them with: pip install -r requirements.txt"
    ) from exc

from gesture_classifier import (
    UNKNOWN,
    GestureSmoother,
    classify_gesture,
    landmarks_to_points,
)

# Initialize Flask
app = Flask(__name__)

# Top-level variables for WSGI / ASGI / Vercel Serverless Function entrypoints
application = app
handler = app

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Hand Gesture Recognition</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #0b0f19;
            color: #f1f5f9;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px 15px;
        }
        .container {
            width: 100%;
            max-width: 840px;
            background: #151c2c;
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            border: 1px solid #1e293b;
        }
        header { text-align: center; margin-bottom: 22px; }
        h1 { font-size: 28px; font-weight: 700; color: #ffffff; margin-bottom: 8px; }
        .subtitle { color: #94a3b8; font-size: 15px; }
        .badge {
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 10px;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            border-bottom: 1px solid #334155;
            padding-bottom: 10px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: #94a3b8;
            padding: 10px 18px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 10px;
            transition: all 0.2s;
        }
        .tab-btn.active {
            background: #2563eb;
            color: #ffffff;
        }
        .tab-btn:hover:not(.active) {
            background: #1e293b;
            color: #e2e8f0;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        /* Camera Viewport */
        .video-wrapper {
            position: relative;
            width: 100%;
            height: 440px;
            background: #090d16;
            border-radius: 16px;
            overflow: hidden;
            border: 2px solid #334155;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1); /* Mirror view */
        }
        video.unmirrored {
            transform: scaleX(1);
        }
        .camera-placeholder {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            color: #64748b;
            text-align: center;
            padding: 20px;
        }
        .camera-placeholder svg { width: 64px; height: 64px; stroke: #475569; }
        
        /* Controls */
        .controls {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 18px;
        }
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 11px 22px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }
        .btn-primary { background: #10b981; color: white; }
        .btn-primary:hover { background: #059669; }
        .btn-danger { background: #ef4444; color: white; display: none; }
        .btn-danger:hover { background: #dc2626; }
        .btn-secondary { background: #334155; color: #f1f5f9; }
        .btn-secondary:hover { background: #475569; }
        
        /* Gesture Display Card */
        .gesture-card {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        }
        .gesture-label-title {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .gesture-value {
            font-size: 28px;
            font-weight: 700;
            color: #38bdf8;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        .gesture-value.detected { color: #34d399; }
        .gesture-value.waiting { color: #94a3b8; font-size: 20px; font-weight: 500; }
        
        /* Supported gestures badges */
        .supported-gestures {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 16px;
        }
        .gesture-tag {
            background: #1e293b;
            color: #cbd5e1;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 13px;
            border: 1px solid #334155;
        }

        /* Upload Area */
        .upload-dropzone {
            border: 2px dashed #475569;
            border-radius: 16px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            background: #090d16;
            transition: all 0.2s;
        }
        .upload-dropzone:hover {
            border-color: #38bdf8;
            background: #0e1626;
        }
        .preview-img {
            max-width: 100%;
            max-height: 350px;
            border-radius: 12px;
            margin-top: 20px;
            display: none;
        }

        .api-info {
            background: #090d16;
            border-radius: 10px;
            padding: 14px;
            margin-top: 22px;
            font-family: monospace;
            font-size: 13px;
            color: #94a3b8;
            border: 1px solid #1e293b;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI Hand Gesture Recognition</h1>
            <p class="subtitle">Real-time computer vision recognition running on MediaPipe</p>
            <span class="badge">● Live & Ready</span>
        </header>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('camera')">🎥 Live Camera</button>
            <button class="tab-btn" onclick="switchTab('upload')">📁 Upload Photo</button>
        </div>

        <!-- CAMERA TAB -->
        <div id="cameraTab" class="tab-content active">
            <div class="video-wrapper">
                <video id="webcam" autoplay playsinline muted></video>
                <div id="cameraPlaceholder" class="camera-placeholder">
                    <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                    <p style="font-size: 17px; font-weight: 600; color: #cbd5e1;">Camera is currently stopped</p>
                    <p style="font-size: 14px;">Click the <strong>Start Camera</strong> button below to open your camera</p>
                </div>
            </div>

            <div class="controls">
                <button id="startBtn" class="btn btn-primary" onclick="startCamera()">▶ Start Camera</button>
                <button id="stopBtn" class="btn btn-danger" onclick="stopCamera()">⏹ Stop Camera</button>
                <button id="flipBtn" class="btn btn-secondary" onclick="toggleMirror()">🔄 Flip Mirror</button>
                <button id="snapBtn" class="btn btn-secondary" onclick="captureOnce()">📸 Single Detect</button>
            </div>

            <div class="gesture-card">
                <div class="gesture-label-title">Detected Gesture</div>
                <div id="gestureDisplay" class="gesture-value waiting">
                    <span>🔍 Start camera to recognize gestures</span>
                </div>
            </div>
        </div>

        <!-- UPLOAD PHOTO TAB -->
        <div id="uploadTab" class="tab-content">
            <div class="upload-dropzone" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleImageUpload(event)">
                <p style="font-size: 18px; font-weight: 600; margin-bottom: 6px;">Click or drag & drop a hand photo here</p>
                <p style="color: #64748b; font-size: 14px;">Supports JPG, PNG, WEBP</p>
            </div>
            <div style="text-align: center;">
                <img id="imagePreview" class="preview-img" alt="Preview">
            </div>
            <div class="gesture-card" style="margin-top: 15px;">
                <div class="gesture-label-title">Photo Gesture Result</div>
                <div id="uploadGestureDisplay" class="gesture-value waiting">
                    <span>Select an image to analyze</span>
                </div>
            </div>
        </div>

        <div class="supported-gestures">
            <span class="gesture-tag">✋ Open Hand</span>
            <span class="gesture-tag">✊ Fist</span>
            <span class="gesture-tag">👍 Thumbs Up</span>
            <span class="gesture-tag">✌️ Victory / Peace</span>
        </div>

        <div class="api-info">
            • <strong>API Endpoints:</strong> <code>GET /health</code> | <code>POST /predict</code><br>
            • <strong>Camera Note:</strong> Allow browser camera permission when prompted.
        </div>
    </div>

    <!-- Hidden canvas for capturing video frames -->
    <canvas id="hiddenCanvas" style="display:none;"></canvas>

    <script>
        const video = document.getElementById('webcam');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const placeholder = document.getElementById('cameraPlaceholder');
        const gestureDisplay = document.getElementById('gestureDisplay');
        const canvas = document.getElementById('hiddenCanvas');
        const ctx = canvas.getContext('2d');

        let cameraStream = null;
        let isDetecting = false;
        let detectionInterval = null;
        let isMirrored = true;

        const EMOJI_MAP = {
            'Open Hand': '✋',
            'Fist': '✊',
            'Thumbs Up': '👍',
            'Victory / Peace': '✌️',
            'No hand detected': '🔍',
            'Unknown': '❓'
        };

        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            if (tab === 'camera') {
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('cameraTab').classList.add('active');
            } else {
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('uploadTab').classList.add('active');
            }
        }

        async function startCamera() {
            try {
                gestureDisplay.className = 'gesture-value waiting';
                gestureDisplay.innerHTML = '<span>Requesting camera permission...</span>';

                cameraStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    },
                    audio: false
                });

                video.srcObject = cameraStream;
                await video.play();

                placeholder.style.display = 'none';
                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-flex';

                gestureDisplay.innerHTML = '<span>Camera ready! Show your hand...</span>';

                // Start continuous detection loop every 300ms
                if (detectionInterval) clearInterval(detectionInterval);
                detectionInterval = setInterval(sendFrameForPrediction, 300);

            } catch (err) {
                console.error("Camera access error:", err);
                placeholder.style.display = 'flex';
                gestureDisplay.className = 'gesture-value waiting';
                gestureDisplay.innerHTML = `<span style="color:#f87171;">⚠️ Camera access denied: ${err.message}</span>`;
                alert("Please allow camera permission in your browser address bar to use the live camera.");
            }
        }

        function stopCamera() {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
            }
            if (detectionInterval) {
                clearInterval(detectionInterval);
                detectionInterval = null;
            }
            video.srcObject = null;
            placeholder.style.display = 'flex';
            startBtn.style.display = 'inline-flex';
            stopBtn.style.display = 'none';

            gestureDisplay.className = 'gesture-value waiting';
            gestureDisplay.innerHTML = '<span>Camera stopped</span>';
        }

        function toggleMirror() {
            isMirrored = !isMirrored;
            video.className = isMirrored ? '' : 'unmirrored';
        }

        async function sendFrameForPrediction() {
            if (!cameraStream || isDetecting || video.videoWidth === 0) return;

            isDetecting = true;
            try {
                // Resize for fast serverless inference
                canvas.width = 480;
                canvas.height = Math.round((480 / video.videoWidth) * video.videoHeight);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                const base64Data = canvas.toDataURL('image/jpeg', 0.7);

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data })
                });

                const data = await response.json();
                if (data.status === 'success') {
                    const label = data.gesture;
                    const emoji = EMOJI_MAP[label] || '👋';
                    gestureDisplay.className = 'gesture-value detected';
                    gestureDisplay.innerHTML = `<span>${emoji}</span> <span>${label}</span>`;
                }
            } catch (err) {
                console.warn("Prediction frame dropped:", err);
            } finally {
                isDetecting = false;
            }
        }

        function captureOnce() {
            sendFrameForPrediction();
        }

        async function handleImageUpload(e) {
            const file = e.target.files[0];
            if (!file) return;

            const preview = document.getElementById('imagePreview');
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';

            const display = document.getElementById('uploadGestureDisplay');
            display.className = 'gesture-value waiting';
            display.innerHTML = '<span>Analyzing uploaded photo...</span>';

            const formData = new FormData();
            formData.append('image', file);

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok && data.status === 'success') {
                    const emoji = EMOJI_MAP[data.gesture] || '👋';
                    display.className = 'gesture-value detected';
                    display.innerHTML = `<span>${emoji}</span> <span>${data.gesture}</span>`;
                } else {
                    display.className = 'gesture-value waiting';
                    display.innerHTML = `<span style="color:#f87171;">⚠️ ${data.message || 'Error detecting'}</span>`;
                }
            } catch (err) {
                display.className = 'gesture-value waiting';
                display.innerHTML = `<span style="color:#f87171;">⚠️ Failed to connect to API</span>`;
            }
        }
    </script>
</body>
</html>
"""


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    accept = request.headers.get("Accept", "")
    if "text/html" in accept and "application/json" not in accept and not request.args.get("json"):
        return render_template_string(HTML_PAGE)

    return jsonify({
        "status": "success",
        "project": "AI Hand Gesture Recognition",
        "message": "Gesture Recognition API is running",
        "available_gestures": [
            "Open Hand",
            "Fist",
            "Thumbs Up",
            "Victory / Peace"
        ],
        "endpoints": {
            "home": "/",
            "health": "/health",
            "predict": "/predict"
        }
    })


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "API is working properly"
    })


# Gesture prediction endpoint (Supports JSON Base64 and Multipart Form-Data)
@app.route("/predict", methods=["POST"])
def predict():
    image_bytes = None

    # Handle JSON payload (base64 image from web camera feed)
    if request.is_json:
        body = request.get_json(silent=True) or {}
        img_str = body.get("image", "")
        if not img_str:
            return jsonify({
                "status": "error",
                "message": "Missing 'image' in JSON body"
            }), 400
        if "," in img_str:
            img_str = img_str.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(img_str)
        except Exception:
            return jsonify({
                "status": "error",
                "message": "Invalid base64 image data"
            }), 400

    # Handle multipart form-data file upload
    elif "image" in request.files:
        file = request.files["image"]
        if not file.filename:
            return jsonify({
                "status": "error",
                "message": "No image selected"
            }), 400
        image_bytes = file.read()

    else:
        return jsonify({
            "status": "error",
            "message": "Please upload an image using the 'image' field (form-data or JSON base64)"
        }), 400

    if not image_bytes:
        return jsonify({
            "status": "error",
            "message": "Uploaded image is empty"
        }), 400

    try:
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "status": "error",
                "message": "Invalid image format"
            }), 400

        height, width = frame.shape[:2]

        # Detect hand landmarks using MediaPipe
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                return jsonify({
                    "status": "success",
                    "gesture": "No hand detected",
                    "message": "Please show your hand clearly in front of the camera"
                })

            hand = results.multi_hand_landmarks[0]
            points = landmarks_to_points(hand.landmark, width, height)
            gesture = classify_gesture(points)

        return jsonify({
            "status": "success",
            "gesture": gesture,
            "message": "Gesture detected successfully"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def run_desktop_webcam(camera_index: int = 0, mirror: bool = True):
    """Run native desktop OpenCV camera window with MediaPipe overlay."""
    print(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        print("Please verify your webcam is plugged in and camera access is allowed.")
        return

    smoother = GestureSmoother(size=7)
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:
        print("Webcam started! Press 'q' or 'Esc' in the camera window to quit, 's' to save screenshot.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab camera frame.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_label = "No hand detected"
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                points = landmarks_to_points(hand.landmark, w, h)
                raw_gesture = classify_gesture(points)
                gesture_label = smoother.update(raw_gesture)
            else:
                smoother.reset()

            # HUD Display
            cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Gesture: {gesture_label}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("AI Hand Gesture Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                os.makedirs("screenshots", exist_ok=True)
                import datetime

                filename = f"screenshots/screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved to {filename}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Hand Gesture Recognition")
    parser.add_argument(
        "--desktop",
        "--webcam",
        action="store_true",
        help="Run native OpenCV desktop camera window directly",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for local web server (default: 5000)",
    )
    args = parser.parse_args()

    if args.desktop:
        run_desktop_webcam(camera_index=args.camera)
    else:
        url = f"http://localhost:{args.port}"
        print("=" * 60)
        print("AI Hand Gesture Recognition App Starting!")
        print(f"• Live Camera Web App: {url}")
        print("• Opening browser with live camera feed automatically...")
        print("• (Tip: To run native OpenCV desktop window instead, run: python main.py --desktop)")
        print("=" * 60)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        app.run(host="0.0.0.0", port=args.port, debug=False)
