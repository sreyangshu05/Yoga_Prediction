import os
import cv2  # type: ignore
import json  # For reading Poses.json
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
import requests  # type: ignore 

# Load dataset
DATASET_PATH = "yoga_data/Poses.json"

# Parse the JSON file
with open(DATASET_PATH, "r") as file:
    pose_data = json.load(file)

# Initialize data and labels
data = []
labels = []

# Directory to save downloaded images
os.makedirs("downloaded_images", exist_ok=True)

# Iterate through the JSON data
for entry in pose_data["Poses"]:  # Access the "Poses" array
    img_url = entry["img_url"]  # Update based on the JSON structure
    label = entry["english_name"]  # Use 'english_name' as label

    try:
        # Download the image
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            # Save the image locally
            img_path = os.path.join("downloaded_images", f"{label}.jpg")
            with open(img_path, "wb") as f:
                f.write(response.content)

            # Load and preprocess the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Image at {img_url} could not be decoded. Skipping.")
                continue

            img = cv2.resize(img, (128, 128))
            data.append(img)
            labels.append(label)
        else:
            print(f"Failed to download image at {img_url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image at {img_url}: {e}")

# Normalize image data
data = np.array(data) / 255.0
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")  # Dynamic output size
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model
os.makedirs("model", exist_ok=True)
model.save("model/yoga_model.h5")

print("Model trained and saved successfully!")