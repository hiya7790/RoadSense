"""
models/evaluate.py — Model Evaluation for VisionSuspend
Generates classification report and confusion matrix.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import argparse

CLASSES = ["smooth", "gravel", "pothole", "wet"]
IMG_SIZE = (224, 224)


def evaluate(model_path, data_dir, batch_size=32):
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    gen = ImageDataGenerator(rescale=1.0 / 255)
    data = gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    preds = model.predict(data, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = data.classes

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — VisionSuspend")
    plt.tight_layout()
    plt.savefig("models/saved/confusion_matrix.png")
    print("Confusion matrix saved to models/saved/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/saved/mobilenetv2_best.h5")
    parser.add_argument("--data_dir", type=str, default="data/augmented")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.model, args.data_dir, args.batch_size)
