"""
AI Hand Gesture Recognition
Vercel-compatible Flask App with Client-Side MediaPipe AI Tracking.

Architecture:
    - Runs lightweight (< 15 MB) on Vercel, avoiding [LAMBDA_SIZE_EXCEEDED] limits.
    - MediaPipe Hands runs client-side in the browser via WebAssembly for 60 FPS real-time tracking.
    - Server provides REST API (/health, /predict) and serves the interactive application.
    - Local desktop OpenCV mode available via: python main.py --desktop
"""

import argparse
import base64
import os
import sys
import tempfile
import webbrowser

# Serverless cache configuration
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from flask import Flask, jsonify, render_template_string, request

# Optional local CV2 and MediaPipe dependencies for desktop execution
try:
    import cv2
    import mediapipe as mp
    import numpy as np

    HAS_LOCAL_CV = True
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
except ImportError:
    cv2 = None
    mp = None
    np = None
    HAS_LOCAL_CV = False

from gesture_classifier import (
    UNKNOWN,
    GestureSmoother,
    classify_gesture,
    landmarks_to_points,
)

# Initialize Flask
app = Flask(__name__)

# Vercel / WSGI / ASGI entrypoint aliases
application = app
handler = app

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Hand Gesture Recognition</title>
    <!-- MediaPipe Hands & Camera Dependencies (Client-side WebAssembly) -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #090d16;
            color: #f1f5f9;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px 15px;
        }
        .container {
            width: 100%;
            max-width: 860px;
            background: #131b2e;
            border-radius: 20px;
            padding: 26px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            border: 1px solid #1e293b;
        }
        header { text-align: center; margin-bottom: 20px; }
        h1 { font-size: 28px; font-weight: 700; color: #ffffff; margin-bottom: 6px; }
        .subtitle { color: #94a3b8; font-size: 14px; }
        .badge {
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 8px;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin: 18px 0;
            border-bottom: 1px solid #334155;
            padding-bottom: 10px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: #94a3b8;
            padding: 9px 18px;
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
        
        /* Video & Canvas Viewport */
        .viewport-wrapper {
            position: relative;
            width: 100%;
            height: 460px;
            background: #050811;
            border-radius: 16px;
            overflow: hidden;
            border: 2px solid #334155;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* Hidden raw video used as camera source */
        video#webcam {
            display: none;
        }
        /* Output canvas showing camera + drawn landmarks */
        canvas#outputCanvas {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1); /* Default mirror view */
        }
        canvas#outputCanvas.unmirrored {
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
            z-index: 5;
        }
        .camera-placeholder svg { width: 60px; height: 60px; stroke: #475569; }
        
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
            padding: 10px 22px;
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
            background: #0b0f19;
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        }
        .gesture-label-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .gesture-value {
            font-size: 26px;
            font-weight: 700;
            color: #38bdf8;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            min-height: 40px;
        }
        .gesture-value.detected { color: #34d399; }
        .gesture-value.waiting { color: #94a3b8; font-size: 18px; font-weight: 500; }
        
        /* Supported gestures badges */
        .supported-gestures {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 18px;
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
            background: #050811;
            transition: all 0.2s;
        }
        .upload-dropzone:hover {
            border-color: #38bdf8;
            background: #0e1626;
        }
        .preview-canvas {
            max-width: 100%;
            max-height: 400px;
            border-radius: 12px;
            margin-top: 20px;
            display: none;
            border: 1px solid #334155;
        }

        .api-info {
            background: #0b0f19;
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
            <p class="subtitle">Real-time MediaPipe Hand Tracking & Gesture Classification</p>
            <span class="badge">● Vercel Serverless Ready (< 15 MB)</span>
        </header>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('camera')">🎥 Live Camera</button>
            <button class="tab-btn" onclick="switchTab('upload')">📁 Upload Photo</button>
        </div>

        <!-- CAMERA TAB -->
        <div id="cameraTab" class="tab-content active">
            <div class="viewport-wrapper">
                <video id="webcam" playsinline muted></video>
                <canvas id="outputCanvas" width="640" height="480"></canvas>
                <div id="cameraPlaceholder" class="camera-placeholder">
                    <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                    <p style="font-size: 17px; font-weight: 600; color: #cbd5e1;">Camera is paused</p>
                    <p style="font-size: 14px;">Click the <strong>Start Camera</strong> button below to open your camera</p>
                </div>
            </div>

            <div class="controls">
                <button id="startBtn" class="btn btn-primary" onclick="startCamera()">▶ Start Camera</button>
                <button id="stopBtn" class="btn btn-danger" onclick="stopCamera()">⏹ Stop Camera</button>
                <button id="flipBtn" class="btn btn-secondary" onclick="toggleMirror()">🔄 Flip Mirror</button>
            </div>

            <div class="gesture-card">
                <div class="gesture-label-title">Detected Gesture</div>
                <div id="gestureDisplay" class="gesture-value waiting">
                    <span>🔍 Start camera to detect hand gestures</span>
                </div>
            </div>
        </div>

        <!-- UPLOAD PHOTO TAB -->
        <div id="uploadTab" class="tab-content">
            <div class="upload-dropzone" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleImageUpload(event)">
                <p style="font-size: 18px; font-weight: 600; margin-bottom: 6px;">Click or drag & drop a hand photo here</p>
                <p style="color: #64748b; font-size: 14px;">Analyzed instantly in your browser</p>
            </div>
            <div style="text-align: center;">
                <canvas id="uploadCanvas" class="preview-canvas"></canvas>
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
            • <strong>Backend API:</strong> <code>GET /health</code> | <code>POST /predict</code><br>
            • <strong>Note:</strong> Allow camera permission when your browser prompts.
        </div>
    </div>

    <script>
        const videoElement = document.getElementById('webcam');
        const canvasElement = document.getElementById('outputCanvas');
        const canvasCtx = canvasElement.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const placeholder = document.getElementById('cameraPlaceholder');
        const gestureDisplay = document.getElementById('gestureDisplay');

        let cameraInstance = null;
        let handsInstance = null;
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

        // Distance formula
        function dist(a, b) {
            return Math.hypot(a.x - b.x, a.y - b.y);
        }

        // Rule-based classification matching gesture_classifier.py
        function classifyGesture(landmarks) {
            if (!landmarks || landmarks.length !== 21) return 'No hand detected';

            const wrist = landmarks[0];
            const fingers = {
                index: dist(landmarks[8], wrist) > dist(landmarks[6], wrist) * 1.1,
                middle: dist(landmarks[12], wrist) > dist(landmarks[10], wrist) * 1.1,
                ring: dist(landmarks[16], wrist) > dist(landmarks[14], wrist) * 1.1,
                pinky: dist(landmarks[20], wrist) > dist(landmarks[18], wrist) * 1.1
            };

            const pinkyBase = landmarks[17];
            const thumbExtended = dist(landmarks[4], pinkyBase) > dist(landmarks[3], pinkyBase) * 1.05;

            // Thumb pointing up check
            const mcp = landmarks[2];
            const tip = landmarks[4];
            const dx = tip.x - mcp.x;
            const dy = tip.y - mcp.y;
            const handSize = dist(landmarks[0], landmarks[9]);
            const longEnough = Math.hypot(dx, dy) > 0.5 * handSize;
            const mostlyVertical = -dy >= Math.abs(dx);
            const aboveKnuckles = tip.y < landmarks[5].y;
            const thumbPointsUp = dy < 0 && longEnough && mostlyVertical && aboveKnuckles;

            const fourExtended = fingers.index && fingers.middle && fingers.ring && fingers.pinky;
            const fourCurled = !fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky;

            if (fourExtended) return 'Open Hand';
            if (fingers.index && fingers.middle && !fingers.ring && !fingers.pinky) return 'Victory / Peace';
            if (fourCurled) {
                if (thumbExtended) return thumbPointsUp ? 'Thumbs Up' : 'Unknown';
                return 'Fist';
            }
            return 'Unknown';
        }

        function initMediaPipe() {
            if (handsInstance) return handsInstance;

            handsInstance = new Hands({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
            });

            handsInstance.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            handsInstance.onResults(onResults);
            return handsInstance;
        }

        function onResults(results) {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                const landmarks = results.multiHandLandmarks[0];

                // Draw hand landmarks overlay
                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF88', lineWidth: 4 });
                drawLandmarks(canvasCtx, landmarks, { color: '#FF0055', lineWidth: 2, radius: 4 });

                const gesture = classifyGesture(landmarks);
                const emoji = EMOJI_MAP[gesture] || '👋';
                gestureDisplay.className = 'gesture-value detected';
                gestureDisplay.innerHTML = `<span>${emoji}</span> <span>${gesture}</span>`;
            } else {
                gestureDisplay.className = 'gesture-value waiting';
                gestureDisplay.innerHTML = '<span>🔍 Point your hand clearly at the camera</span>';
            }
            canvasCtx.restore();
        }

        async function startCamera() {
            try {
                placeholder.style.display = 'none';
                gestureDisplay.className = 'gesture-value waiting';
                gestureDisplay.innerHTML = '<span>Starting camera & loading AI model...</span>';

                const hands = initMediaPipe();

                cameraInstance = new Camera(videoElement, {
                    onFrame: async () => {
                        await hands.send({ image: videoElement });
                    },
                    width: 640,
                    height: 480
                });

                await cameraInstance.start();

                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-flex';
                gestureDisplay.innerHTML = '<span>Camera ready! Show your hand...</span>';

            } catch (err) {
                console.error("Camera start error:", err);
                placeholder.style.display = 'flex';
                gestureDisplay.className = 'gesture-value waiting';
                gestureDisplay.innerHTML = `<span style="color:#f87171;">⚠️ Camera error: ${err.message}</span>`;
                alert("Please allow camera permission in your browser address bar.");
            }
        }

        function stopCamera() {
            if (cameraInstance) {
                cameraInstance.stop();
                cameraInstance = null;
            }
            if (videoElement.srcObject) {
                videoElement.srcObject.getTracks().forEach(t => t.stop());
                videoElement.srcObject = null;
            }
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            placeholder.style.display = 'flex';
            startBtn.style.display = 'inline-flex';
            stopBtn.style.display = 'none';
            gestureDisplay.className = 'gesture-value waiting';
            gestureDisplay.innerHTML = '<span>Camera stopped</span>';
        }

        function toggleMirror() {
            isMirrored = !isMirrored;
            canvasElement.className = isMirrored ? '' : 'unmirrored';
        }

        // Photo upload analysis
        async function handleImageUpload(e) {
            const file = e.target.files[0];
            if (!file) return;

            const uploadCanvas = document.getElementById('uploadCanvas');
            const uploadCtx = uploadCanvas.getContext('2d');
            const display = document.getElementById('uploadGestureDisplay');

            display.className = 'gesture-value waiting';
            display.innerHTML = '<span>Analyzing photo with AI...</span>';

            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = async () => {
                uploadCanvas.width = img.width;
                uploadCanvas.height = img.height;
                uploadCanvas.style.display = 'block';

                const hands = new Hands({
                    locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
                });
                hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.5 });

                hands.onResults((res) => {
                    uploadCtx.drawImage(img, 0, 0, uploadCanvas.width, uploadCanvas.height);
                    if (res.multiHandLandmarks && res.multiHandLandmarks.length > 0) {
                        const lms = res.multiHandLandmarks[0];
                        drawConnectors(uploadCtx, lms, HAND_CONNECTIONS, { color: '#00FF88', lineWidth: 4 });
                        drawLandmarks(uploadCtx, lms, { color: '#FF0055', lineWidth: 2, radius: 4 });
                        const g = classifyGesture(lms);
                        display.className = 'gesture-value detected';
                        display.innerHTML = `<span>${EMOJI_MAP[g] || '👋'}</span> <span>${g}</span>`;
                    } else {
                        display.className = 'gesture-value waiting';
                        display.innerHTML = '<span>No hand detected in photo</span>';
                    }
                });

                await hands.send({ image: img });
            };
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
            "Victory / Peace",
        ],
        "endpoints": {
            "home": "/",
            "health": "/health",
            "predict": "/predict",
        },
    })


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "API is working properly",
    })


# Gesture prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Handle JSON input with 21 landmarks coordinates
    if request.is_json:
        data = request.get_json(silent=True) or {}

        # Landmarks format: [{"x": 0.5, "y": 0.5}, ...] or [[0.5, 0.5], ...]
        if "landmarks" in data:
            raw_pts = data["landmarks"]
            pts = []
            for p in raw_pts:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
                elif isinstance(p, dict) and "x" in p and "y" in p:
                    pts.append((float(p["x"]), float(p["y"])))

            if len(pts) == 21:
                gesture = classify_gesture(pts)
                return jsonify({
                    "status": "success",
                    "gesture": gesture,
                    "message": "Gesture classified successfully from landmarks",
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Expected 21 landmarks, received {len(pts)}",
                }), 400

        # Base64 image in JSON
        img_str = data.get("image", "")
        if img_str:
            if not HAS_LOCAL_CV:
                return jsonify({
                    "status": "error",
                    "message": "Serverless runtime uses client-side MediaPipe. Send 'landmarks' array or use the web UI.",
                }), 400
            if "," in img_str:
                img_str = img_str.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(img_str)
            except Exception:
                return jsonify({
                    "status": "error",
                    "message": "Invalid base64 image data",
                }), 400
            return _process_image_bytes(image_bytes)

        return jsonify({
            "status": "error",
            "message": "Missing 'landmarks' or 'image' field in JSON request",
        }), 400

    # 2. Handle multipart form-data image file upload
    elif "image" in request.files:
        if not HAS_LOCAL_CV:
            return jsonify({
                "status": "error",
                "message": "Serverless runtime uses client-side MediaPipe. Send 'landmarks' array or use the web UI.",
            }), 400

        file = request.files["image"]
        if not file.filename:
            return jsonify({
                "status": "error",
                "message": "No image selected",
            }), 400
        image_bytes = file.read()
        return _process_image_bytes(image_bytes)

    return jsonify({
        "status": "error",
        "message": "Please provide 'landmarks' JSON or upload an image",
    }), 400


def _process_image_bytes(image_bytes):
    """Process raw image bytes using OpenCV and MediaPipe (local environment)."""
    if not image_bytes:
        return jsonify({"status": "error", "message": "Image is empty"}), 400

    try:
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        h, w = frame.shape[:2]
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        ) as hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                return jsonify({
                    "status": "success",
                    "gesture": "No hand detected",
                    "message": "Please show hand clearly",
                })

            hand = results.multi_hand_landmarks[0]
            points = landmarks_to_points(hand.landmark, w, h)
            gesture = classify_gesture(points)

        return jsonify({
            "status": "success",
            "gesture": gesture,
            "message": "Gesture detected successfully",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def run_desktop_webcam(camera_index: int = 0, mirror: bool = True):
    """Run native desktop OpenCV camera window with MediaPipe overlay."""
    if not HAS_LOCAL_CV:
        print("Error: Desktop webcam mode requires opencv and mediapipe.")
        print("Install them with: pip install -r requirements-local.txt")
        return

    print(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    smoother = GestureSmoother(size=7)
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        print("Webcam started! Press 'q' or 'Esc' in camera window to quit, 's' to save screenshot.")
        while True:
            ret, frame = cap.read()
            if not ret:
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

            # HUD
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
