# CNN Digit Classifier with OpenCV Drawing Demo

This project implements a convolutional neural network (CNN) that classifies 20×20 grayscale digit images (0–9).  
It includes:

- A **training script** with **K-fold cross-validation** and **data augmentation** (translation, scale, rotation, stroke-thickness changes).
- An **interactive OpenCV GUI** that lets you draw digits with the mouse and get real-time predictions from the trained model.

The code is written in Python with PyTorch and is designed for a small custom digit dataset extracted from a sprite sheet (`digits.png`).

---

## Features

- **Convolutional neural network** for 10-digit classification.
- **Stratified K-fold cross-validation** (e.g., K=10 or 12).
- **Data augmentation**:
  - Random translation (jitter position).
  - Random scaling and small rotations.
  - Optional stroke thickness variation via morphological operations (erode/dilate).
- **Train/test split** with balanced classes.
- **Model checkpoint saving** (e.g., `crossvalidate_9.pyt`).
- **OpenCV drawing app**:
  - Draw with the mouse on a canvas.
  - Canvas is downsampled to 20×20 and normalized to match training.
  - Live prediction shown in the window title and printed to console.

---

## Project Structure



```text
.
├── digits.png                     # 1000×2000 sprite of digits (20×20 tiles)
├── convolutional-crossvalidate.py # training script (CNN + K-fold + augmentation)
├── draw_digits.py                 # OpenCV drawing + live prediction
├── crossvalidate_9.pyt            # saved model weights (created after training)
└── README.md
