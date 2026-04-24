"""
RoadSense — Kaggle Training Notebook
=====================================
Upload this as a Kaggle notebook with GPU enabled.

SETUP:
1. Upload your data/raw/ folder as a Kaggle Dataset named "roadsense-rtk"
2. Create a new Kaggle Notebook
3. Add the dataset to the notebook
4. Set Accelerator to GPU T4 x2 (Settings → Accelerator)
5. Paste this entire script and Run All

OUTPUT:
- custom_cnn_best.h5      (Custom CNN - best val accuracy)
- custom_cnn_final.h5     (Custom CNN - final epoch)
- mobilenetv2_best.h5     (MobileNetV2 - best val accuracy)
- mobilenetv2_final.h5    (MobileNetV2 - final epoch)
- training_comparison.png (Accuracy/loss plots for both models)
- evaluation_report.txt   (Classification reports)
- confusion_matrices.png  (Side-by-side confusion matrices)

Download the .h5 files and place them in models/saved/ on your laptop.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, GlobalAveragePooling2D, Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "/kaggle/working"
PREPARED_DIR = os.path.join(OUTPUT_DIR, "prepared_data")
CLASSES = ["smooth", "gravel", "pothole", "wet"]
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 25

# ============================================================================
# STEP 0: Find and prepare dataset (RTK format -> our 4 classes)
# ============================================================================

# RTK class mapping -> our classes
RTK_MAPPING = {
    "asphaltGood": "smooth",
    "asphaltRegular": "smooth",
    "asphaltBad": "pothole",
    "pavedRegular": "gravel",
    "pavedBad": "pothole",
    "unpavedRegular": "gravel",
    "unpavedBad": "pothole",
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def find_rtk_folders(search_root="/kaggle/input/datasets/tallwinkingstan/road-traversing-knowledge-rtk-dataset"):
    """Find all RTK class folders (asphaltGood, pavedBad, etc.) under search root."""
    found = {}
    print(f"\nScanning {search_root} for RTK dataset folders...")
    for root, dirs, files in os.walk(search_root):
        for d in dirs:
            if d in RTK_MAPPING:
                full_path = os.path.join(root, d)
                img_count = len([f for f in os.listdir(full_path)
                                if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS])
                if img_count > 0:
                    found[d] = full_path
                    print(f"  Found: {d} -> {RTK_MAPPING[d]} ({img_count} images)")
    
    # Also check if data is already in smooth/gravel/pothole/wet format
    for root, dirs, files in os.walk(search_root):
        if set(CLASSES).issubset(set(dirs)):
            print(f"\n  Dataset already in prepared format at: {root}")
            return None, root  # Already prepared

    if not found:
        # Print full tree for debugging
        print("\nERROR: No RTK folders found. Directory tree:")
        for root, dirs, files in os.walk(search_root):
            depth = root.replace(search_root, "").count(os.sep)
            if depth > 4:
                continue
            indent = "  " * depth
            print(f"{indent}{os.path.basename(root)}/ ({len(files)} files, {len(dirs)} dirs)")
        raise FileNotFoundError("Could not find RTK dataset folders.")

    return found, None


def prepare_dataset():
    """Map RTK folders to our 4 classes and generate synthetic wet images."""
    found_folders, already_prepared = find_rtk_folders()

    if already_prepared:
        print(f"Using pre-prepared dataset at: {already_prepared}")
        return already_prepared

    # Create output structure
    for cls in CLASSES:
        os.makedirs(os.path.join(PREPARED_DIR, cls), exist_ok=True)

    # Copy and reorganize
    import shutil
    class_counters = {c: 0 for c in CLASSES}

    print("\nReorganizing dataset...")
    for rtk_name, src_dir in found_folders.items():
        target_class = RTK_MAPPING[rtk_name]
        files = [f for f in os.listdir(src_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

        for fname in files:
            src_path = os.path.join(src_dir, fname)
            # Validate image
            img = cv2.imread(src_path)
            if img is None or img.shape[0] < 10 or img.shape[1] < 10:
                continue

            class_counters[target_class] += 1
            ext = os.path.splitext(fname)[1].lower()
            new_name = f"{target_class}_{class_counters[target_class]:05d}{ext}"
            dst_path = os.path.join(PREPARED_DIR, target_class, new_name)
            shutil.copy2(src_path, dst_path)

        print(f"  {rtk_name} -> {target_class}: {len(files)} images")

    # Generate synthetic wet images from smooth
    import albumentations as A
    wet_aug = A.Compose([
        A.RandomRain(brightness_coefficient=0.8, drop_width=2, blur_value=3, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.0), p=0.8),
    ])

    smooth_dir = os.path.join(PREPARED_DIR, "smooth")
    wet_dir = os.path.join(PREPARED_DIR, "wet")
    smooth_files = [f for f in os.listdir(smooth_dir)
                    if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

    num_wet = min(400, len(smooth_files))
    print(f"\n  Generating {num_wet} synthetic wet images...")
    for i in range(num_wet):
        fname = smooth_files[i % len(smooth_files)]
        img = cv2.imread(os.path.join(smooth_dir, fname))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = wet_aug(image=img_rgb)["image"]
        cv2.imwrite(
            os.path.join(wet_dir, f"wet_{i+1:05d}.jpg"),
            cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR),
        )

    # Summary
    print("\nPrepared dataset summary:")
    for cls in CLASSES:
        cls_dir = os.path.join(PREPARED_DIR, cls)
        count = len(os.listdir(cls_dir))
        print(f"  {cls}: {count} images")

    return PREPARED_DIR


import cv2
DATA_DIR = prepare_dataset()
print(f"\nDATA_DIR = {DATA_DIR}")

# ============================================================================
# DATA GENERATORS (80/20 split, on-the-fly augmentation for training)
# ============================================================================

print("\n" + "=" * 60)
print("  Setting up data generators...")
print("=" * 60)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    classes=CLASSES,
    shuffle=True,
)

val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    classes=CLASSES,
    shuffle=False,
)

print(f"Training samples:   {train_data.samples}")
print(f"Validation samples: {val_data.samples}")

# ============================================================================
# MODEL 1: CUSTOM CNN FROM SCRATCH
# ============================================================================

print("\n" + "=" * 60)
print("  MODEL 1: Custom CNN (from scratch)")
print("=" * 60)


def build_custom_cnn():
    model = Sequential([
        # Block 1: edges, simple textures (224 -> 112)
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(224, 224, 3)),
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
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


custom_model = build_custom_cnn()
custom_model.summary()

custom_callbacks = [
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "custom_cnn_best.h5"),
        save_best_only=True, monitor="val_accuracy", verbose=1,
    ),
    EarlyStopping(patience=7, monitor="val_loss", restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
]

print("\nTraining Custom CNN...")
custom_history = custom_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=custom_callbacks,
)

custom_model.save(os.path.join(OUTPUT_DIR, "custom_cnn_final.h5"))
print("Custom CNN training complete!")

# ============================================================================
# MODEL 2: MOBILENETV2 (Fine-tuned)
# ============================================================================

print("\n" + "=" * 60)
print("  MODEL 2: MobileNetV2 (fine-tuned)")
print("=" * 60)

# Reset generators (they track position)
train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    classes=CLASSES,
    shuffle=True,
)
val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    classes=CLASSES,
    shuffle=False,
)


def build_mobilenet(fine_tune_layers=20):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

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
        metrics=["accuracy"],
    )
    return model


mobilenet_model = build_mobilenet()
print(f"MobileNetV2 total params: {mobilenet_model.count_params():,}")

mobilenet_callbacks = [
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "mobilenetv2_best.h5"),
        save_best_only=True, monitor="val_accuracy", verbose=1,
    ),
    EarlyStopping(patience=7, monitor="val_loss", restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
]

print("\nTraining MobileNetV2...")
mobilenet_history = mobilenet_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=mobilenet_callbacks,
)

mobilenet_model.save(os.path.join(OUTPUT_DIR, "mobilenetv2_final.h5"))
print("MobileNetV2 training complete!")

# ============================================================================
# TRAINING COMPARISON PLOTS
# ============================================================================

print("\n" + "=" * 60)
print("  Generating comparison plots...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RoadSense — Training Comparison", fontsize=16, fontweight="bold")

# Custom CNN accuracy
axes[0, 0].plot(custom_history.history["accuracy"], label="Train", linewidth=2)
axes[0, 0].plot(custom_history.history["val_accuracy"], label="Validation", linewidth=2)
axes[0, 0].set_title("Custom CNN — Accuracy")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Custom CNN loss
axes[0, 1].plot(custom_history.history["loss"], label="Train", linewidth=2)
axes[0, 1].plot(custom_history.history["val_loss"], label="Validation", linewidth=2)
axes[0, 1].set_title("Custom CNN — Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# MobileNetV2 accuracy
axes[1, 0].plot(mobilenet_history.history["accuracy"], label="Train", linewidth=2)
axes[1, 0].plot(mobilenet_history.history["val_accuracy"], label="Validation", linewidth=2)
axes[1, 0].set_title("MobileNetV2 — Accuracy")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# MobileNetV2 loss
axes[1, 1].plot(mobilenet_history.history["loss"], label="Train", linewidth=2)
axes[1, 1].plot(mobilenet_history.history["val_loss"], label="Validation", linewidth=2)
axes[1, 1].set_title("MobileNetV2 — Loss")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_comparison.png"), dpi=150)
plt.show()
print("Training comparison plot saved!")

# ============================================================================
# EVALUATION — Classification Reports & Confusion Matrices
# ============================================================================

print("\n" + "=" * 60)
print("  Evaluating both models...")
print("=" * 60)

# Fresh validation generator (no augmentation, no shuffle)
eval_gen = ImageDataGenerator(rescale=1.0 / 255)
eval_data = eval_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    shuffle=False,
)

report_lines = []


def evaluate_model(model, model_name, eval_data):
    """Evaluate model and return predictions."""
    eval_data.reset()
    preds = model.predict(eval_data, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = eval_data.classes

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    cm = confusion_matrix(y_true, y_pred)

    report_lines.append(f"\n{'='*50}")
    report_lines.append(f"  {model_name} — Classification Report")
    report_lines.append(f"{'='*50}")
    report_lines.append(report)

    print(f"\n{model_name} Classification Report:")
    print(report)

    return cm


# Evaluate Custom CNN (load best checkpoint)
custom_best = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, "custom_cnn_best.h5"))
custom_cm = evaluate_model(custom_best, "Custom CNN", eval_data)

# Evaluate MobileNetV2 (load best checkpoint)
mobilenet_best = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, "mobilenetv2_best.h5"))
mobilenet_cm = evaluate_model(mobilenet_best, "MobileNetV2", eval_data)

# Save reports
with open(os.path.join(OUTPUT_DIR, "evaluation_report.txt"), "w") as f:
    f.write("\n".join(report_lines))
print("\nEvaluation report saved!")

# Confusion matrices side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RoadSense — Confusion Matrices", fontsize=14, fontweight="bold")

sns.heatmap(custom_cm, annot=True, fmt="d", xticklabels=CLASSES,
            yticklabels=CLASSES, cmap="Blues", ax=ax1)
ax1.set_title("Custom CNN")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("True")

sns.heatmap(mobilenet_cm, annot=True, fmt="d", xticklabels=CLASSES,
            yticklabels=CLASSES, cmap="Greens", ax=ax2)
ax2.set_title("MobileNetV2")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
plt.show()
print("Confusion matrices saved!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)

custom_val_acc = max(custom_history.history["val_accuracy"])
mobilenet_val_acc = max(mobilenet_history.history["val_accuracy"])

print(f"\n  Custom CNN    — Best Val Accuracy: {custom_val_acc*100:.2f}%")
print(f"  MobileNetV2   — Best Val Accuracy: {mobilenet_val_acc*100:.2f}%")
print(f"\n  Files saved to {OUTPUT_DIR}:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f:40s} ({size/1024/1024:.1f} MB)")

print(f"\n  Download the .h5 files and place in models/saved/ on your laptop.")
print(f"  Then run: python server.py --model models/saved/custom_cnn_best.h5")
print("=" * 60)
