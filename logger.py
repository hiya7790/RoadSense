"""
logger.py — Session Data Logger for RoadSense
Logs every prediction to a timestamped CSV file for post-analysis.
"""

import os
import csv
import time
from datetime import datetime


class SessionLogger:
    """Logs predictions to CSV files in the logs/ directory."""

    FIELDNAMES = [
        "timestamp", "datetime", "road_class", "confidence",
        "prob_smooth", "prob_gravel", "prob_pothole", "prob_wet",
        "suspension_mode", "stiffness", "damping",
        "fps", "latency_ms", "model_type",
    ]

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"session_{session_time}.csv")
        self.start_time = time.time()
        self.total_predictions = 0
        self.class_counts = {"smooth": 0, "gravel": 0, "pothole": 0, "wet": 0}

        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def log(self, prediction_data: dict):
        """Log a single prediction row."""
        row = {
            "timestamp": prediction_data.get("timestamp", time.time()),
            "datetime": datetime.now().isoformat(),
            "road_class": prediction_data.get("class", ""),
            "confidence": prediction_data.get("confidence", 0),
            "prob_smooth": prediction_data.get("probabilities", {}).get("smooth", 0),
            "prob_gravel": prediction_data.get("probabilities", {}).get("gravel", 0),
            "prob_pothole": prediction_data.get("probabilities", {}).get("pothole", 0),
            "prob_wet": prediction_data.get("probabilities", {}).get("wet", 0),
            "suspension_mode": prediction_data.get("suspension_mode", ""),
            "stiffness": prediction_data.get("stiffness", 0),
            "damping": prediction_data.get("damping", 0),
            "fps": prediction_data.get("fps", 0),
            "latency_ms": prediction_data.get("latency_ms", 0),
            "model_type": prediction_data.get("model_type", ""),
        }
        self._writer.writerow(row)
        self._file.flush()
        self.total_predictions += 1
        cls = row["road_class"]
        if cls in self.class_counts:
            self.class_counts[cls] += 1

    def get_stats(self) -> dict:
        """Return session statistics."""
        duration = time.time() - self.start_time
        total = max(self.total_predictions, 1)
        return {
            "session_file": self.filepath,
            "duration_seconds": round(duration, 1),
            "total_predictions": self.total_predictions,
            "class_distribution": {
                k: round(v / total * 100, 1) for k, v in self.class_counts.items()
            },
        }

    def close(self):
        """Close the CSV file."""
        if self._file and not self._file.closed:
            self._file.close()
