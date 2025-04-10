import cv2
import numpy as np
import joblib

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === Load Trained Classifier ===
model = joblib.load("yoga_pose_classifier.pkl")

# === Mediapipe Pose Setup ===
model_path = "pose_landmarker_heavy.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# === Webcam Pose Classification ===
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    try:
        result = detector.detect(mp_image)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            feature_vector = []
            for lm in landmarks:
                feature_vector.extend([lm.x, lm.y, lm.z])

            # Classify the pose
            prediction = model.predict([feature_vector])[0]

            # Draw label on screen
            cv2.putText(frame, f'Pose: {prediction}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Detection error:", e)

    cv2.imshow("Yoga Pose Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
