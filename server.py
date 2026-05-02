"""
server.py — FastAPI Server for RoadSense
REST API + WebSocket for real-time road surface classification.

Swagger docs available at: http://localhost:8000/docs

Run with:
    python server.py
    python server.py --model models/saved/custom_cnn_best.h5
    python server.py --camera 1 --port 8080
"""

import os
import asyncio
import argparse
import time
import json
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from engine import InferenceEngine
from logger import SessionLogger


# ── Pydantic Models (for Swagger docs) ───────────────────────────────────────

class StatusResponse(BaseModel):
    camera_active: bool = Field(description="Whether camera is capturing frames")
    model_loaded: bool = Field(description="Whether a model is loaded")
    model_type: str = Field(description="Active model type: custom, mobilenet, or none")
    model_path: str = Field(description="Path to the active model file")
    gradcam_enabled: bool = Field(description="Whether GradCAM overlay is active")
    smoothing_window: int = Field(description="Temporal smoothing window size")
    fps: float = Field(description="Current inference FPS")
    latency_ms: float = Field(description="Per-frame inference latency in ms")
    session_duration: float = Field(description="Seconds since engine started")
    running: bool = Field(description="Whether the engine is running")


class ProbabilityBreakdown(BaseModel):
    smooth: float = Field(ge=0, le=1, description="Probability of smooth road")
    gravel: float = Field(ge=0, le=1, description="Probability of gravel road")
    pothole: float = Field(ge=0, le=1, description="Probability of pothole")
    wet: float = Field(ge=0, le=1, description="Probability of wet road")


class RawPrediction(BaseModel):
    road_class: str = Field(alias="class", description="Predicted road surface class")
    confidence: float = Field(description="Confidence of the prediction")
    probabilities: ProbabilityBreakdown = Field(description="Per-class probabilities")

    class Config:
        populate_by_name = True


class SuspensionState(BaseModel):
    mode: str = Field(description="Suspension mode: Soft, Medium, Firm, or Adaptive")
    stiffness: float = Field(description="Spring stiffness in N/m")
    damping: float = Field(description="Damping coefficient in Ns/m")
    recommended_action: str = Field(description="Recommended driver action")


class MetricsInfo(BaseModel):
    fps: float = Field(description="Current frames per second")
    latency_ms: float = Field(description="Inference latency in milliseconds")
    model_type: str = Field(description="Active model type")


class PredictionResponse(BaseModel):
    timestamp: float = Field(description="Unix timestamp of the prediction")
    raw: RawPrediction = Field(description="Single-frame raw prediction")
    smoothed: RawPrediction = Field(description="Temporally smoothed prediction")
    suspension: SuspensionState = Field(description="Recommended suspension state")
    metrics: MetricsInfo = Field(description="Performance metrics")


class ModelInfo(BaseModel):
    name: str = Field(description="Model filename")
    type: str = Field(description="Model type: custom or mobilenet")
    path: str = Field(description="File path to the model")
    active: bool = Field(description="Whether this model is currently active")


class SwitchModelRequest(BaseModel):
    model_path: str = Field(description="Path to the model file to load")
    model_type: str = Field(
        default="auto",
        description="Model type: custom, mobilenet, or auto (infer from filename)",
    )


class GradCAMToggleRequest(BaseModel):
    enabled: bool = Field(description="Enable or disable GradCAM overlay")


class SmoothingRequest(BaseModel):
    window_size: int = Field(ge=1, le=30, description="Smoothing window size (1-30 frames)")


class SessionStats(BaseModel):
    session_file: str = Field(description="Path to the session CSV log file")
    duration_seconds: float = Field(description="Session duration in seconds")
    total_predictions: int = Field(description="Total predictions made")
    class_distribution: dict = Field(description="Percentage time per road class")


class HistoryEntry(BaseModel):
    timestamp: float
    road_class: str = Field(alias="class")
    confidence: float

    class Config:
        populate_by_name = True


# ── Global State ─────────────────────────────────────────────────────────────

engine: Optional[InferenceEngine] = None
logger: Optional[SessionLogger] = None
startup_args = {}


# ── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, logger

    engine = InferenceEngine(
        camera_id=startup_args.get("camera", 0),
        smoothing_window=startup_args.get("smoothing", 8),
    )

    model_path = startup_args.get("model")
    if model_path and os.path.exists(model_path):
        engine.load_model(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print("No model loaded. Use POST /api/models/switch to load one.")

    engine.start()
    logger = SessionLogger()
    print(f"Session log: {logger.filepath}")

    yield

    engine.stop()
    logger.close()


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RoadSense API",
    description=(
        "Real-time road surface classification and adaptive suspension system.\n\n"
        "## Features\n"
        "- Live camera inference with CNN classification\n"
        "- Temporal smoothing for stable predictions\n"
        "- GradCAM model explainability overlays\n"
        "- Suspension parameter recommendations\n"
        "- Model hot-swap (Custom CNN / MobileNetV2)\n"
        "- WebSocket real-time streaming\n\n"
        "## WebSocket\n"
        "Connect to `ws://localhost:8000/ws/live` for real-time prediction stream."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Dashboard Static Files ───────────────────────────────────────────────────

dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.isdir(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")


# ── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health_check():
    """Basic health check."""
    return {"status": "ok", "service": "RoadSense"}


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get full engine status including camera, model, and metrics."""
    return engine.get_status()


# ── Predictions ──────────────────────────────────────────────────────────────

@app.get("/api/prediction", response_model=PredictionResponse, tags=["Predictions"])
async def get_prediction():
    """
    Get the latest prediction result.

    Returns the raw (single-frame) prediction, the temporally smoothed prediction,
    recommended suspension parameters, and performance metrics.
    """
    result = engine.get_latest()

    # Log to session file
    if logger and result.smoothed_class:
        logger.log({
            "timestamp": result.timestamp,
            "class": result.smoothed_class,
            "confidence": result.smoothed_confidence,
            "probabilities": result.smoothed_probabilities,
            "suspension_mode": result.suspension_mode,
            "stiffness": result.stiffness,
            "damping": result.damping,
            "fps": result.fps,
            "latency_ms": result.latency_ms,
            "model_type": result.model_type,
        })

    return result.to_dict()


@app.get("/api/prediction/history", tags=["Predictions"])
async def get_prediction_history(
    seconds: int = Query(default=60, ge=1, le=300, description="How many seconds of history"),
):
    """Get recent prediction history for timeline visualization."""
    return engine.get_prediction_history(seconds)


# ── Suspension ───────────────────────────────────────────────────────────────

@app.get("/api/suspension", response_model=SuspensionState, tags=["Suspension"])
async def get_suspension():
    """
    Get the current recommended suspension state.

    Returns the mode (Soft/Medium/Firm/Adaptive), spring stiffness (N/m),
    damping coefficient (Ns/m), and a recommended driver action.
    """
    result = engine.get_latest()
    return {
        "mode": result.suspension_mode or "Unknown",
        "stiffness": result.stiffness,
        "damping": result.damping,
        "recommended_action": result.recommended_action or "Waiting for prediction...",
    }


# ── Camera / Frames ─────────────────────────────────────────────────────────

@app.get("/api/frame", tags=["Camera"])
async def get_frame(
    gradcam: bool = Query(default=False, description="Overlay GradCAM heatmap"),
):
    """
    Get the latest camera frame as a JPEG image.

    Set gradcam=true to overlay the GradCAM heatmap on the frame.
    Can be used directly in an <img> tag.
    """
    jpeg = engine.get_frame_jpeg(with_gradcam=gradcam)
    if jpeg is None:
        raise HTTPException(status_code=503, detail="No frame available. Is the camera connected?")
    return StreamingResponse(
        iter([jpeg]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/frame/base64", tags=["Camera"])
async def get_frame_base64(
    gradcam: bool = Query(default=False, description="Overlay GradCAM heatmap"),
):
    """Get the latest camera frame as a base64-encoded JPEG string."""
    b64 = engine.get_frame_base64(with_gradcam=gradcam)
    if b64 is None:
        raise HTTPException(status_code=503, detail="No frame available.")
    return {"frame": b64, "encoding": "base64", "format": "jpeg"}


@app.get("/api/stream/video", tags=["Camera"])
async def video_stream():
    """
    MJPEG video stream endpoint.

    Use this directly in an <img> tag:
        <img src="http://localhost:8000/api/stream/video" />

    Streams at approximately 15 FPS.
    """
    async def generate():
        while True:
            jpeg = engine.get_frame_jpeg(with_gradcam=engine._gradcam_enabled)
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.066)  # ~15 FPS

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Models ───────────────────────────────────────────────────────────────────

@app.get("/api/models", response_model=list[ModelInfo], tags=["Models"])
async def list_models():
    """List all available models in models/saved/ directory."""
    return engine.get_available_models()


@app.get("/api/models/active", tags=["Models"])
async def get_active_model():
    """Get information about the currently active model."""
    return {
        "model_type": engine.model_type,
        "model_path": engine.model_path,
        "loaded": engine.model is not None,
    }


@app.post("/api/models/switch", tags=["Models"])
async def switch_model(req: SwitchModelRequest):
    """
    Switch the active model.

    Hot-swaps the model without restarting the engine. Clears the
    temporal smoothing buffer on switch.
    """
    try:
        engine.load_model(req.model_path, req.model_type)
        return {
            "success": True,
            "message": f"Switched to {req.model_path}",
            "model_type": engine.model_type,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Configuration ────────────────────────────────────────────────────────────

@app.get("/api/config", tags=["Configuration"])
async def get_config():
    """Get current engine configuration."""
    return {
        "gradcam_enabled": engine._gradcam_enabled,
        "smoothing_window": engine.smoothing_window,
        "camera_id": engine.camera_id,
    }


@app.post("/api/config/gradcam", tags=["Configuration"])
async def toggle_gradcam(req: GradCAMToggleRequest):
    """Enable or disable GradCAM overlay computation."""
    engine.set_gradcam(req.enabled)
    return {"gradcam_enabled": engine._gradcam_enabled}


@app.post("/api/config/smoothing", tags=["Configuration"])
async def set_smoothing(req: SmoothingRequest):
    """Set the temporal smoothing window size."""
    engine.set_smoothing_window(req.window_size)
    return {"smoothing_window": engine.smoothing_window}


# ── Session ──────────────────────────────────────────────────────────────────

@app.get("/api/session/stats", response_model=SessionStats, tags=["Session"])
async def get_session_stats():
    """Get current session statistics (duration, class distribution, etc.)."""
    if logger is None:
        raise HTTPException(status_code=503, detail="Logger not initialized.")
    return logger.get_stats()


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Real-time WebSocket stream.

    Sends JSON messages at ~15 FPS containing:
    - prediction data (raw + smoothed)
    - suspension parameters
    - performance metrics
    - camera frame (base64 JPEG, if requested)

    Send a JSON message to configure:
        {"include_frame": true}   — include base64 frames
        {"include_frame": false}  — prediction data only (lower bandwidth)
        {"gradcam": true/false}   — toggle GradCAM
    """
    await ws.accept()
    include_frame = False

    try:
        while True:
            # Check for incoming config messages (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                try:
                    config = json.loads(msg)
                    if "include_frame" in config:
                        include_frame = bool(config["include_frame"])
                    if "gradcam" in config:
                        engine.set_gradcam(bool(config["gradcam"]))
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                pass

            # Build response
            result = engine.get_latest()
            data = result.to_dict()

            if include_frame:
                b64 = engine.get_frame_base64(with_gradcam=engine._gradcam_enabled)
                data["frame"] = b64

            await ws.send_json(data)

            # Log
            if logger and result.smoothed_class:
                logger.log({
                    "timestamp": result.timestamp,
                    "class": result.smoothed_class,
                    "confidence": result.smoothed_confidence,
                    "probabilities": result.smoothed_probabilities,
                    "suspension_mode": result.suspension_mode,
                    "stiffness": result.stiffness,
                    "damping": result.damping,
                    "fps": result.fps,
                    "latency_ms": result.latency_ms,
                    "model_type": result.model_type,
                })

            await asyncio.sleep(0.066)  # ~15 FPS

    except WebSocketDisconnect:
        pass


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    global startup_args

    parser = argparse.ArgumentParser(description="RoadSense API Server")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (.h5)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--smoothing", type=int, default=8,
                        help="Temporal smoothing window")

    args = parser.parse_args()
    startup_args = vars(args)

    print(f"\n{'='*50}")
    print(f"  RoadSense API Server")
    print(f"  Dashboard:    http://localhost:{args.port}/dashboard/")
    print(f"  Swagger docs: http://localhost:{args.port}/docs")
    print(f"  WebSocket:    ws://localhost:{args.port}/ws/live")
    print(f"  Video stream: http://localhost:{args.port}/api/stream/video")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
