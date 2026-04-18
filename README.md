# 🚗 RoadSense — Intelligent Road Surface Classification for Adaptive Suspension

> A real-time computer vision system that classifies road surface conditions and recommends optimal vehicle suspension settings using deep learning and interactive simulation.

---

## 📌 Project Overview

It is an intelligent, vision-based system that classifies road surface conditions in real time and recommends optimal suspension settings for improved vehicle comfort and safety. The system leverages a camera feed as input and applies a Convolutional Neural Network (CNN) to identify road types, enabling adaptive suspension responses.

The dataset consists of four road surface categories — **smooth asphalt**, **gravel**, **pothole**, and **wet road** — with approximately 200–300 images per class collected from publicly available datasets such as the RTK dataset, supplemented with curated images from online sources. To enhance model robustness and generalization, extensive data augmentation is applied using the **Albumentations** library, including brightness variation, motion blur, flipping, and noise injection.

The model uses **MobileNetV2** pretrained on ImageNet, fine-tuned for this four-class classification task. This architecture is chosen for its lightweight design and efficiency, making it suitable for real-time inference on standard laptops. Each predicted class maps to a suspension setting:

| Road Surface   | Suspension Setting |
|----------------|--------------------|
| Smooth Asphalt | Soft               |
| Gravel         | Medium             |
| Pothole        | Firm               |
| Wet Road       | Adaptive           |

A simulation layer built with **Pygame** provides an interactive dashboard displaying the live video feed with classification labels overlaid, a dynamic suspension stiffness dial that updates based on predictions, and a confidence bar indicating model certainty. The system mimics real-world adaptive suspension behavior in a simplified virtual environment.

VisionSuspend demonstrates the integration of computer vision, deep learning, and real-time simulation to address a practical automotive problem, serving as a strong foundation for future work in intelligent vehicle systems.

---

## 🗂️ Project Structure

```
visionsuspend/
├── data/
│   ├── raw/                  # Original collected images
│   │   ├── smooth/
│   │   ├── gravel/
│   │   ├── pothole/
│   │   └── wet/
│   └── augmented/            # Augmented dataset output
├── models/
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Evaluation & metrics
│   └── saved/                # Saved model checkpoints (.h5 / .pt)
├── simulation/
│   └── dashboard.py          # Pygame simulation dashboard
├── inference.py              # Real-time inference from webcam
├── augment.py                # Data augmentation pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/hiya7790/visionsuspend.git
cd visionsuspend
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the model
```bash
python models/train.py --data_dir data/raw --epochs 20 --batch_size 32
```

### Run augmentation pipeline
```bash
python augment.py --input data/raw --output data/augmented
```

### Run real-time inference (webcam)
```bash
python inference.py --model models/saved/mobilenetv2_best.h5
```

### Launch Pygame simulation dashboard
```bash
python simulation/dashboard.py --model models/saved/mobilenetv2_best.h5
```

---

## 🧠 Model Details

- **Architecture:** MobileNetV2 (pretrained on ImageNet)
- **Fine-tuning:** Last few layers unfrozen for task-specific learning
- **Input size:** 224 × 224 RGB
- **Output:** 4-class softmax (smooth, gravel, pothole, wet)
- **Framework:** TensorFlow / Keras

---

## 🎮 Simulation Dashboard Features

- Live video feed with real-time classification overlay
- Dynamic suspension stiffness dial (Soft / Medium / Firm / Adaptive)
- Confidence bar showing model certainty per prediction
- Color-coded road condition alerts

---

## 📦 Dependencies

See `requirements.txt` for the full list. Key libraries:

- `tensorflow` / `keras` — model training and inference
- `albumentations` — data augmentation
- `opencv-python` — video capture and frame processing
- `pygame` — interactive simulation dashboard
- `numpy`, `matplotlib`, `scikit-learn` — utilities and evaluation

---

## 🔮 Future Work

- Deploy on edge hardware (Raspberry Pi / Jetson Nano)
- Expand dataset with more road types (snow, cobblestone)
- Integrate GPS-based road mapping
- Real CAN bus integration with vehicle suspension ECU

---

## 📄 License

This project is licensed under the MIT License.
