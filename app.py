from flask import Flask, render_template, Response, request # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Load the trained model
model = load_model("model/yoga_model.h5")
categories = ["Category1", "Category2", "Category3"]  # Replace with actual categories

# Video capture function
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the frame
        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, 128, 128, 3))

        # Predict pose
        prediction = model.predict(reshaped_frame)
        class_index = np.argmax(prediction)
        class_label = categories[class_index]

        # Display prediction on frame
        cv2.putText(frame, f"Pose: {class_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/feedback')
def feedback():
    # Dummy feedback example for detected pose
    detected_pose = "Warrior Pose"  # Replace with dynamic detection
    feedback = "Your alignment looks good. Ensure your back leg is fully extended and your arms are parallel to the ground."

    return render_template("results.html", detected_pose=detected_pose, feedback=feedback)


if __name__ == '__main__':
    app.run(debug=True)
