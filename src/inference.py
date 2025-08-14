import tensorflow as tf
import cv2
import numpy as np
import sys
from src.dataset import get_data_generators  # to get class labels

# Load model
MODEL_PATH = "models/best_vgg16_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Get class labels
train_generator, _, _ = get_data_generators(batch_size=1)
class_labels = list(train_generator.class_indices.keys())

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess_face(face_img):
    """Resize and normalize the face for model prediction."""
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=(0, -1))  # shape: (1, 48, 48, 1)
    return face_img


def predict_image(img_path):
    """Predict emotion from an image file."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        preprocessed = preprocess_face(face_img)
        prediction = model.predict(preprocessed)
        class_idx = np.argmax(prediction)
        emotion = class_labels[class_idx]
        print(f"Detected emotion: {emotion}")

    if len(faces) == 0:
        print("No face detected.")


def run_webcam():
    """Run real-time emotion detection with face detection."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            preprocessed = preprocess_face(face_img)
            prediction = model.predict(preprocessed)
            class_idx = np.argmax(prediction)
            emotion = class_labels[class_idx]

            # Draw rectangle & label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_image(img_path)
    else:
        print("Starting webcam. Press 'q' to quit.")
        run_webcam()
