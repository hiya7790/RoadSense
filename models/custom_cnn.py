"""
models/custom_cnn.py — Custom CNN Architecture for RoadSense
A purpose-built lightweight CNN for 4-class road surface classification.

Architecture:
    Conv2D(32)  -> BN -> ReLU -> MaxPool
    Conv2D(64)  -> BN -> ReLU -> MaxPool
    Conv2D(128) -> BN -> ReLU -> MaxPool
    Conv2D(256) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    Conv2D(256) -> BN -> ReLU -> GlobalAvgPool
    Dense(256)  -> ReLU -> Dropout(0.5)
    Dense(128)  -> ReLU -> Dropout(0.3)
    Dense(4, softmax)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, GlobalAveragePooling2D, Dense,
)
from tensorflow.keras.optimizers import Adam

NUM_CLASSES = 4
IMG_SHAPE = (224, 224, 3)


def build_custom_cnn(learning_rate=1e-3):
    """Build and compile the custom CNN."""
    model = Sequential([
        # Block 1: edges, simple textures (224 -> 112)
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=IMG_SHAPE),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2: texture patterns (112 -> 56)
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3: crack patterns, gravel clusters (56 -> 28)
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 4: surface type signatures (28 -> 14)
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 5: final feature extraction (14x14x256 -> 256)
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        # Classifier head
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
