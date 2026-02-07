# Audio Emotion Recognition Project

**Name:** Vihan Singh Parihar
**ID:** 2025AAPS0766P

## 1. Project Overview
This project implements a Convolutional Neural Network (CNN) to classify audio clips into 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

## 2. Model Performance
- **Test Accuracy:** 63%
- **Macro F1-Score:** 0.58
- **Gender Bias Analysis:** The model shows a minimal bias gap (approx 1.5%) between male and female speakers.

## 3. Training Details
- **Architecture:** Custom CNN with 3 Convolutional Blocks (Conv2D -> BatchNorm -> MaxPool -> Dropout).
- **Input Features:** Log-Mel Spectrograms (128 Mel bands x 130 Time steps).
- **Key Settings:**
  - Optimizer: Adam
  - Loss Function: Categorical Crossentropy
  - Batch Size: 32
  - Epochs: 30
- **Data Processing:** Audio files were trimmed of silence and padded/cropped to a fixed duration of 3 seconds.

## 4. How to Run (One-Command Execution)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
