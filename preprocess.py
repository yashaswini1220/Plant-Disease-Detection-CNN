import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================
# DATASET PATH
# =========================================

dataset_path = r"D:\PlantVillage\PlantVillage"

# =========================================
# LISTS
# =========================================

data = []
labels = []

# =========================================
# IMAGE SETTINGS
# =========================================

IMG_SIZE = 128

image_extensions = ('.jpg', '.jpeg', '.png')

# =========================================
# LOAD DATASET
# =========================================

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        for image_name in os.listdir(folder_path):

            if image_name.lower().endswith(image_extensions):

                image_path = os.path.join(
                    folder_path,
                    image_name
                )

                # Read image
                img = cv2.imread(image_path)

                if img is not None:

                    # Resize image
                    img = cv2.resize(
                        img,
                        (IMG_SIZE, IMG_SIZE)
                    )

                    # Normalize image
                    img = img / 255.0

                    # Store data
                    data.append(img)

                    labels.append(folder)

# =========================================
# CONVERT TO NUMPY ARRAYS
# =========================================

data = np.array(data)

labels = np.array(labels)

# =========================================
# DATASET INFO
# =========================================

print("Dataset Loaded Successfully")

print("Total Images:", len(data))

print("Data Shape:", data.shape)

print("Labels Shape:", labels.shape)

# =========================================
# LABEL ENCODING
# =========================================

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

# Save class names
np.save("class_names.npy", encoder.classes_)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)

print("Testing Shape:", X_test.shape)

# =========================================
# DATA AUGMENTATION
# =========================================

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

print("Data Augmentation Applied Successfully")

# =========================================
# CNN MODEL
# =========================================

model = Sequential()

# First Convolution Layer
model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(128,128,3)
))

# First Pooling Layer
model.add(MaxPooling2D(
    pool_size=(2,2)
))

# Second Convolution Layer
model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

# Second Pooling Layer
model.add(MaxPooling2D(
    pool_size=(2,2)
))

# Flatten Layer
model.add(Flatten())

# Dense Layer
model.add(Dense(
    128,
    activation='relu'
))

# Dropout Layer
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(
    len(np.unique(labels)),
    activation='softmax'
))

# =========================================
# COMPILE MODEL
# =========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================
# MODEL SUMMARY
# =========================================

model.summary()

# =========================================
# EARLY STOPPING
# =========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# =========================================
# TRAINING
# =========================================

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =========================================
# SAVE MODEL
# =========================================

model.save("plant_disease_model.keras")

print("Model Saved Successfully")