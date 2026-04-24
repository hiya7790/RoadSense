"""
engine.py — Threaded Real-Time Inference Engine for RoadSense

Features:
    - Separate camera and inference threads (no frame drops)
    - Temporal smoothing (exponential moving average over N frames)
    - GradCAM heatmap generation for model explainability
    - FPS and latency tracking
    - Model hot-swap (custom CNN <-> MobileNetV2)
"""

import os
import time
import threading
import base64
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict

import cv2
import numpy as np
import tensorflow as tf


# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = ["smooth", "gravel", "pothole", "wet"]
IMG_SIZE = (224, 224)

SUSPENSION_PARAMS = {
    "smooth":  {"mode": "Soft",     "stiffness": 2000,  "damping": 300,  "action": "Comfort mode engaged"},
    "gravel":  {"mode": "Medium",   "stiffness": 5000,  "damping": 800,  "action": "Stabilization active"},
    "pothole": {"mode": "Firm",     "stiffness": 12000, "damping": 2000, "action": "Impact protection — reduce speed"},
    "wet":     {"mode": "Adaptive", "stiffness": 7000,  "damping": 1200, "action": "Traction control — adaptive damping"},
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Container for a single inference result."""
    timestamp: float = 0.0

    # Raw single-frame prediction
    raw_class: str = ""
    raw_confidence: float = 0.0
    raw_probabilities: Dict[str, float] = field(default_factory=dict)

    # Temporally smoothed prediction
    smoothed_class: str = ""
    smoothed_confidence: float = 0.0
    smoothed_probabilities: Dict[str, float] = field(default_factory=dict)

    # Suspension parameters
    suspension_mode: str = ""
    stiffness: float = 0.0
    damping: float = 0.0
    recommended_action: str = ""

    # Metrics
    fps: float = 0.0
    latency_ms: float = 0.0
    model_type: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "raw": {
                "class": self.raw_class,
                "confidence": round(self.raw_confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in self.raw_probabilities.items()},
            },
            "smoothed": {
                "class": self.smoothed_class,
                "confidence": round(self.smoothed_confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in self.smoothed_probabilities.items()},
            },
            "suspension": {
                "mode": self.suspension_mode,
                "stiffness": self.stiffness,
                "damping": self.damping,
                "recommended_action": self.recommended_action,
            },
            "metrics": {
                "fps": self.fps,
                "latency_ms": self.latency_ms,
                "model_type": self.model_type,
            },
        }


# ── GradCAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(model, img_array, class_idx=None):
    """
    Compute Gradient-weighted Class Activation Mapping (GradCAM).

    Args:
        model: Keras model.
        img_array: Preprocessed image (1, 224, 224, 3).
        class_idx: Target class index. None = use predicted class.

    Returns:
        Heatmap as uint8 numpy array (224, 224), or None on failure.
    """
    # Find the last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        return None

    try:
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output],
        )
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        # Global average pooling of gradients → channel weights
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(conv_outputs[0] * weights[0], axis=-1)

        # ReLU + normalize
        cam = tf.nn.relu(cam)
        cam_max = tf.reduce_max(cam)
        if cam_max > 0:
            cam = cam / cam_max

        # Resize to image dimensions
        cam = tf.image.resize(cam[..., tf.newaxis], IMG_SIZE)[..., 0]
        return (cam.numpy() * 255).astype(np.uint8)

    except Exception:
        return None


def apply_gradcam_overlay(frame, heatmap, alpha=0.4):
    """
    Blend a GradCAM heatmap onto a BGR frame.

    Args:
        frame: Original BGR frame (any size).
        heatmap: GradCAM heatmap uint8 (224, 224).
        alpha: Overlay opacity.

    Returns:
        Blended BGR frame (same size as input).
    """
    h, w = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)


# ── Inference Engine ─────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Threaded real-time inference engine.

    - Camera thread captures frames continuously.
    - Inference thread runs model prediction on the latest frame.
    - Main thread reads the latest result via get_latest().
    """

    def __init__(self, camera_id=0, smoothing_window=8):
        self.camera_id = camera_id
        self.smoothing_window = smoothing_window

        # Model
        self.model = None
        self.model_type = "none"
        self.model_path = ""

        # State flags
        self._running = False
        self._gradcam_enabled = False
        self._camera_active = False

        # Latest data (thread-safe via locks)
        self._latest_frame = None
        self._latest_result = InferenceResult()
        self._latest_gradcam = None
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()

        # Temporal smoothing buffer
        self._pred_history = deque(maxlen=smoothing_window)

        # FPS tracking
        self._fps_times = deque(maxlen=30)
        self._start_time = None

        # Prediction history for timeline
        self._prediction_log = deque(maxlen=600)  # ~60s at 10 FPS

    # ── Model Management ─────────────────────────────────────────────────

    def load_model(self, model_path: str, model_type: str = "auto"):
        """
        Load a Keras model from disk.

        Args:
            model_path: Path to .h5 model file.
            model_type: "custom", "mobilenet", or "auto" (infer from filename).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path

        if model_type == "auto":
            if "custom" in model_path.lower():
                self.model_type = "custom"
            else:
                self.model_type = "mobilenet"
        else:
            self.model_type = model_type

        # Clear smoothing buffer on model switch
        self._pred_history.clear()

    def get_available_models(self) -> list:
        """Scan models/saved/ for available model files."""
        saved_dir = os.path.join("models", "saved")
        if not os.path.exists(saved_dir):
            return []

        models = []
        for f in os.listdir(saved_dir):
            if f.endswith((".h5", ".keras")):
                mtype = "custom" if "custom" in f.lower() else "mobilenet"
                models.append({
                    "name": f,
                    "type": mtype,
                    "path": os.path.join(saved_dir, f),
                    "active": os.path.join(saved_dir, f) == self.model_path,
                })
        return models

    # ── Start / Stop ─────────────────────────────────────────────────────

    def start(self):
        """Start camera and inference threads."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()

        cam_thread = threading.Thread(target=self._capture_loop, daemon=True)
        inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
        cam_thread.start()
        inf_thread.start()

    def stop(self):
        """Stop all threads."""
        self._running = False

    # ── Configuration ────────────────────────────────────────────────────

    def set_gradcam(self, enabled: bool):
        self._gradcam_enabled = enabled

    def set_smoothing_window(self, window: int):
        self.smoothing_window = max(1, min(window, 30))
        self._pred_history = deque(
            list(self._pred_history)[-self.smoothing_window:],
            maxlen=self.smoothing_window,
        )

    # ── Data Access ──────────────────────────────────────────────────────

    def get_latest(self) -> InferenceResult:
        """Get the most recent inference result (thread-safe)."""
        with self._result_lock:
            return self._latest_result

    def get_frame_jpeg(self, with_gradcam=False) -> Optional[bytes]:
        """Get the latest frame as JPEG bytes."""
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return None

        if with_gradcam and self._latest_gradcam is not None:
            frame = apply_gradcam_overlay(frame, self._latest_gradcam)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def get_frame_base64(self, with_gradcam=False) -> Optional[str]:
        """Get the latest frame as a base64-encoded JPEG string."""
        jpeg = self.get_frame_jpeg(with_gradcam)
        if jpeg is None:
            return None
        return base64.b64encode(jpeg).decode("utf-8")

    def get_status(self) -> dict:
        """Get engine status."""
        result = self.get_latest()
        return {
            "camera_active": self._camera_active,
            "model_loaded": self.model is not None,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "gradcam_enabled": self._gradcam_enabled,
            "smoothing_window": self.smoothing_window,
            "fps": result.fps,
            "latency_ms": result.latency_ms,
            "session_duration": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "running": self._running,
        }

    def get_prediction_history(self, seconds: int = 60) -> list:
        """Get recent prediction history."""
        cutoff = time.time() - seconds
        return [p for p in self._prediction_log if p["timestamp"] > cutoff]

    # ── Internal Threads ─────────────────────────────────────────────────

    def _preprocess(self, frame):
        """Preprocess a BGR frame for model input."""
        resized = cv2.resize(frame, IMG_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def _capture_loop(self):
        """Continuously capture frames from camera."""
        cap = cv2.VideoCapture(self.camera_id)
        self._camera_active = cap.isOpened()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._latest_frame = frame

        cap.release()
        self._camera_active = False

    def _inference_loop(self):
        """Continuously run inference on the latest frame."""
        while self._running:
            # Wait for a frame and a model
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()

            if frame is None or self.model is None:
                time.sleep(0.05)
                continue

            start = time.time()

            # Preprocess and predict
            input_tensor = self._preprocess(frame)
            preds = self.model(input_tensor, training=False).numpy()[0]

            latency = (time.time() - start) * 1000

            # GradCAM (optional)
            gradcam_heatmap = None
            if self._gradcam_enabled:
                gradcam_heatmap = compute_gradcam(self.model, input_tensor)
                self._latest_gradcam = gradcam_heatmap

            # Raw prediction
            class_idx = int(np.argmax(preds))
            raw_label = CLASSES[class_idx]
            raw_confidence = float(preds[class_idx])

            # Temporal smoothing
            self._pred_history.append(preds)
            smoothed = np.mean(list(self._pred_history), axis=0)
            sm_idx = int(np.argmax(smoothed))
            sm_label = CLASSES[sm_idx]
            sm_confidence = float(smoothed[sm_idx])

            # FPS
            now = time.time()
            self._fps_times.append(now)
            if len(self._fps_times) > 1:
                elapsed = self._fps_times[-1] - self._fps_times[0]
                fps = round((len(self._fps_times) - 1) / max(elapsed, 0.001), 1)
            else:
                fps = 0.0

            # Suspension params
            susp = SUSPENSION_PARAMS[sm_label]

            result = InferenceResult(
                timestamp=now,
                raw_class=raw_label,
                raw_confidence=raw_confidence,
                raw_probabilities={c: float(preds[i]) for i, c in enumerate(CLASSES)},
                smoothed_class=sm_label,
                smoothed_confidence=sm_confidence,
                smoothed_probabilities={c: float(smoothed[i]) for i, c in enumerate(CLASSES)},
                suspension_mode=susp["mode"],
                stiffness=susp["stiffness"],
                damping=susp["damping"],
                recommended_action=susp["action"],
                fps=fps,
                latency_ms=round(latency, 1),
                model_type=self.model_type,
            )

            with self._result_lock:
                self._latest_result = result

            # Add to history log
            self._prediction_log.append({
                "timestamp": now,
                "class": sm_label,
                "confidence": round(sm_confidence, 3),
            })

            # Throttle to ~20 FPS max inference rate
            time.sleep(0.02)
