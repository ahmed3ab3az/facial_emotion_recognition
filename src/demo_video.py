import tensorflow as tf
import cv2
import numpy as np
import sys
import os
import argparse
from src.dataset import get_data_generators

# Load trained model
MODEL_PATH = "models/best_vgg16_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Get class labels
train_generator, _, _ = get_data_generators(batch_size=1)
class_labels = list(train_generator.class_indices.keys())

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(face_img):
    """Resize and normalize the face for model prediction."""
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=(0, -1))  # shape: (1, 48, 48, 1)
    return face_img

def process_video(video_path, output_path="output_demo.mp4"):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # default to 25 if fps=0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            preprocessed = preprocess_face(face_img)
            prediction = model.predict(preprocessed, verbose=0)
            class_idx = np.argmax(prediction)
            emotion = class_labels[class_idx]

            # Draw bounding box & label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition on Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output_demo.mp4", help="Path to save processed video")
    args = parser.parse_args()

    process_video(args.video, args.output)
