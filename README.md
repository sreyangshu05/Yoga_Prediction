# Yoga Pose Detection and Feedback System

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Details](#dataset-details)
- [Model Training](#model-training)
- [Web Application](#web-application)

---

## Introduction

This project implements a Yoga Pose Detection and Feedback System using TensorFlow, Flask, and OpenCV. The system consists of a machine learning model trained to classify yoga poses and a web-based application to provide real-time feedback on poses.

---

## Features

- **Dataset Parsing and Preprocessing**: Extracts yoga pose data from a JSON file and preprocesses it for model training.
- **Image Classification Model**: Uses Convolutional Neural Networks (CNN) to classify yoga poses.
- **Real-Time Feedback**: Provides real-time pose detection and alignment feedback via webcam.
- **Web Application**: Flask-based interface for real-time pose detection and feedback.

---

## Technologies Used

- **Backend**:
  - Python
  - TensorFlow/Keras
  - Flask
- **Frontend**:
  - HTML
  - CSS
  - JavaScript
- **Libraries**:
  - OpenCV
  - NumPy
  - scikit-learn
  - requests
- **Other**:
  - JSON

---

## System Architecture

1. **Dataset Preparation**:
   - Images downloaded from URLs provided in the `Poses.json` file.
   - Images resized and normalized to feed into the CNN model.
2. **Model Training**:
   - Convolutional layers for feature extraction.
   - Fully connected layers for classification.
   - Trained with an 80-20 train-test split.
3. **Web Application**:
   - Webcam integration using OpenCV for real-time predictions.
   - Flask routes for serving live video feed and feedback.

---

## Installation

### Prerequisites

- Python 3.8 or above
- TensorFlow 2.x
- OpenCV
- Flask
- Virtual environment (optional but recommended)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Yoga-Pose-Detection.git
   cd Yoga-Pose-Detection
   python -m venv venv
   venv\Scripts\activate
2. Install dependencies.
3. Place your dataset file (Poses.json) in the yoga_data directory.
4. Train the model:
   ```bash
   python Train.py
5. Start the Flask application:
   ```bash
   python app.py

## Usage:
 - Open your browser and go to http://127.0.0.1:5000/.
 - Live video feed with detected pose overlay.
 - Feedback on pose alignment.
 - Navigate to the live video feed to see the detected pose in real-time.
 - Receive dynamic feedback based on detected poses.

## Dataset Details:
JSON structure:
- Poses: Array of yoga poses.
Each pose contains:
- img_url: URL of the pose image.
- english_name: Name of the pose in English.
Preprocessing:
- Images resized to 128x128 pixels.
- Normalized pixel values between 0 and 1.

## Model Training:

### Model Architecture:
Input: RGB images of shape (128, 128, 3).
Layers: 
- 2D Convolutional layers with ReLU activation.
- MaxPooling layers for down-sampling.
- Fully connected Dense layers for classification.
Output: Softmax activation for multi-class classification.

### Training Parameters:
Optimizer: Adam
Loss function: Categorical Crossentropy
Batch size: 32
Epochs: 10
Validation split: 20%

Saving the Model:
    ```bash
    model/yoga_model.h5


## Web Application:

### Key Flask Routes:
/: Serves the homepage with instructions.
/video_feed: Streams the live webcam feed with pose detection.
/feedback: Displays feedback based on detected pose.

### Feedback Example:
Detected Pose: Warrior Pose
Feedback: "Your alignment looks good. Ensure your back leg is fully extended and your arms are parallel to the ground."