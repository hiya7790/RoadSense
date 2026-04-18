"""
models/train.py — MobileNetV2 Fine-Tuning for VisionSuspend
Trains a 4-class road surface classifier.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

CLASSES = ["smooth", "gravel", "pothole", "wet"]
IMG_SIZE = (224, 224)
NUM_CLASSES = 4


def build_model(fine_tune_layers=20):
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
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_data_generators(data_dir, batch_size):
    train_gen = ImageDataGenerator(
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
    val_data = train_gen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        classes=CLASSES,
        shuffle=False,
    )
    return train_data, val_data


def plot_history(history, save_path="models/saved/training_plot.png"):
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
    model = build_model(fine_tune_layers=args.fine_tune_layers)
    model.summary()

    train_data, val_data = get_data_generators(args.data_dir, args.batch_size)

    callbacks = [
        ModelCheckpoint("models/saved/mobilenetv2_best.h5", save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=7, monitor="val_loss", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save("models/saved/mobilenetv2_final.h5")
    plot_history(history)
    print("Training complete. Model saved to models/saved/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionSuspend Model Training")
    parser.add_argument("--data_dir", type=str, default="data/augmented", help="Augmented data directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune_layers", type=int, default=20)
    args = parser.parse_args()
    main(args)
