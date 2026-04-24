"""
models/train.py — Model Training for RoadSense
Supports both Custom CNN and MobileNetV2 fine-tuning.

Usage:
    python models/train.py --model_type custom --data_dir data/augmented
    python models/train.py --model_type mobilenet --data_dir data/augmented
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add parent directory to path so we can import custom_cnn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.custom_cnn import build_custom_cnn

CLASSES = ["smooth", "gravel", "pothole", "wet"]
IMG_SIZE = (224, 224)
NUM_CLASSES = 4


def build_mobilenet(fine_tune_layers=20, learning_rate=1e-4):
    """Build MobileNetV2 with fine-tuning."""
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze base, unfreeze last N layers
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_data_generators(data_dir, batch_size):
    """Create train/val data generators with on-the-fly augmentation."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        # On-the-fly augmentation (in addition to offline augmentation)
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    val_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_data = train_gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        classes=CLASSES,
        shuffle=True,
    )
    val_data = val_gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        classes=CLASSES,
        shuffle=False,
    )
    return train_data, val_data


def plot_history(history, save_path):
    """Save training history plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title("Accuracy")
    ax1.legend()
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Loss")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")


def main(args):
    os.makedirs("models/saved", exist_ok=True)

    # Build model based on type
    if args.model_type == "custom":
        print("\n=== Building Custom CNN ===")
        model = build_custom_cnn(learning_rate=args.lr)
        model_name = "custom_cnn"
    else:
        print("\n=== Building MobileNetV2 ===")
        model = build_mobilenet(
            fine_tune_layers=args.fine_tune_layers,
            learning_rate=args.lr,
        )
        model_name = "mobilenetv2"

    model.summary()

    train_data, val_data = get_data_generators(args.data_dir, args.batch_size)

    best_path = f"models/saved/{model_name}_best.h5"
    final_path = f"models/saved/{model_name}_final.h5"
    plot_path = f"models/saved/{model_name}_training_plot.png"

    callbacks = [
        ModelCheckpoint(best_path, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=7, monitor="val_loss", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(final_path)
    plot_history(history, plot_path)
    print(f"\nTraining complete!")
    print(f"  Best model:  {best_path}")
    print(f"  Final model: {final_path}")
    print(f"  Plot:        {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSense Model Training")
    parser.add_argument("--model_type", type=str, default="custom",
                        choices=["custom", "mobilenet"],
                        help="Model type: custom or mobilenet")
    parser.add_argument("--data_dir", type=str, default="data/augmented",
                        help="Augmented data directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (1e-3 for custom, 1e-4 for mobilenet)")
    parser.add_argument("--fine_tune_layers", type=int, default=20,
                        help="Layers to unfreeze for MobileNetV2")
    args = parser.parse_args()

    # Auto-adjust learning rate for mobilenet
    if args.model_type == "mobilenet" and args.lr == 1e-3:
        args.lr = 1e-4
        print("Auto-adjusted learning rate to 1e-4 for MobileNetV2")

    main(args)
