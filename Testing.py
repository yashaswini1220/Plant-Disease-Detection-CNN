import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import load_model

# LOAD TRAINED MODEL
model = load_model("plant_disease_model.keras")

# Dataset path
dataset_path = r"D:\PlantVillage\PlantVillage"

# Lists
data = []
labels = []

# Image size
IMG_SIZE = 128

# Allowed image formats
image_extensions = ('.jpg', '.jpeg', '.png')

# READ DATASET
for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        for image_name in os.listdir(folder_path):

            if image_name.lower().endswith(image_extensions):

                image_path = os.path.join(folder_path, image_name)

                img = cv2.imread(image_path)

                if img is not None:

                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                    img = img / 255.0

                    data.append(img)
                    labels.append(folder)

# Convert to arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# EVALUATION
print("\n--- MODEL EVALUATION ---")

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# PREDICTIONS
print("\n--- PREDICTIONS ---")

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

# CONFUSION MATRIX
print("\n--- CONFUSION MATRIX ---")

cm = confusion_matrix(y_test, y_pred)

print(cm)

# CLASSIFICATION REPORT
print("\n--- CLASSIFICATION REPORT ---")

report = classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
)

print(report)