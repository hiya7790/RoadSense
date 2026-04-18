"""
inference.py — Real-Time Webcam Inference for VisionSuspend
Runs road surface classification on live camera feed using OpenCV.
"""

import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model

CLASSES = ["smooth", "gravel", "pothole", "wet"]
SUSPENSION_MAP = {
    "smooth": "Soft",
    "gravel": "Medium",
    "pothole": "Firm",
    "wet": "Adaptive",
}
COLOR_MAP = {
    "smooth": (0, 200, 0),
    "gravel": (255, 165, 0),
    "pothole": (0, 0, 220),
    "wet": (220, 0, 220),
}
IMG_SIZE = (224, 224)


def preprocess(frame):
    resized = cv2.resize(frame, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb / 255.0
    return np.expand_dims(normalized, axis=0)


def draw_overlay(frame, label, confidence, suspension):
    color = COLOR_MAP.get(label, (255, 255, 255))
    h, w = frame.shape[:2]

    # Road class label
    cv2.rectangle(frame, (10, 10), (400, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Road: {label.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Suspension: {suspension}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = 10, h - 40, int((w - 20) * confidence), 20
    cv2.rectangle(frame, (10, h - 45), (w - 10, h - 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h - 45), (10 + bar_w, h - 20), color, -1)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (15, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def run_inference(model_path, camera_id=0):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded. Starting camera...")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        preds = model.predict(input_tensor, verbose=0)[0]
        class_idx = np.argmax(preds)
        label = CLASSES[class_idx]
        confidence = float(preds[class_idx])
        suspension = SUSPENSION_MAP[label]

        frame = draw_overlay(frame, label, confidence, suspension)
        cv2.imshow("VisionSuspend — Real-Time Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionSuspend Real-Time Inference")
    parser.add_argument("--model", type=str, default="models/saved/mobilenetv2_best.h5")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    args = parser.parse_args()
    run_inference(args.model, args.camera)
