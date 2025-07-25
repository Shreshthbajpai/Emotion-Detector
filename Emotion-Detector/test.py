'''
PyPower Projects
Emotion Detection Using AI - Improved Script
'''

import cv2
import numpy as np
from keras.models import load_model  # type: ignore
from keras.preprocessing.image import img_to_array  # type: ignore
import time
import sys

# Load the Haar cascade for face detection
try:
    face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    if face_classifier.empty():
        raise FileNotFoundError("Could not load Haar cascade. Make sure 'haarcascade_frontalface_default.xml' is in the same directory.")
    print("✅ Haar cascade loaded.")
except Exception as e:
    print(f"❌ Error loading Haar cascade: {e}")
    sys.exit(1)

# Load the emotion classification model
try:
    classifier = load_model('./Emotion_Detection.h5')
    print("✅ Emotion detection model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Define emotion labels
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    sys.exit(1)

print("📷 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)  # add channel dimension
            roi = np.expand_dims(roi, axis=0)   # add batch dimension

            preds = classifier.predict(roi, verbose=0)[0]
            label = class_labels[np.argmax(preds)]
            confidence = np.max(preds)
            print(f"😃 Prediction: {label} ({confidence*100:.2f}%)")

            label_position = (x, y - 10)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()