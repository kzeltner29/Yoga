import os
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === Set up Mediapipe Pose Landmarker ===
model_path = "pose_landmarker_heavy.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False  # not needed for classification
)
detector = vision.PoseLandmarker.create_from_options(options)

# === Landmark Extraction ===
def extract_landmarks_from_folder(image_folder):
    X, y = [], []
    pose_labels = os.listdir(image_folder)

    for label in pose_labels:
        label_path = os.path.join(image_folder, label)
        if not os.path.isdir(label_path):
            continue

        print(f"Processing pose: {label}")
        for img_file in os.listdir(label_path):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to load {img_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                try:
                    result = detector.detect(mp_image)
                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        feature_vector = []
                        for lm in landmarks:
                            feature_vector.extend([lm.x, lm.y, lm.z])
                        X.append(feature_vector)
                        y.append(label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return np.array(X), np.array(y)

# === Training Script ===
def train_model(X, y, output_model_path="yoga_pose_classifier.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, output_model_path)
    print(f"\nModel saved to: {output_model_path}")

# === Main ===
if __name__ == "__main__":
    train_folder = "yoga_dataset"  # Update this if your folder is named differently
    print("Extracting landmarks from images...")
    X, y = extract_landmarks_from_folder(train_folder)

    if len(X) == 0:
        print("No landmarks found. Check your image folder and Mediapipe setup.")
    else:
        print(f"Extracted {len(X)} pose samples.")
        train_model(X, y)
