# 🌿 Plant Disease Detection Using CNN

## 📌 Project Overview

Plant Disease Detection Using Convolutional Neural Networks (CNN) is a deep learning-based system that identifies diseases in plant leaves from uploaded images. The model is trained on the PlantVillage dataset and can classify diseases in Tomato, Potato, and Bell Pepper plants.

The project provides a user-friendly Streamlit web application where users can upload a leaf image and receive disease predictions along with confidence scores and recommendations.

---

## 🎯 Objectives

* Detect plant diseases automatically from leaf images.
* Reduce dependency on manual disease diagnosis.
* Provide fast and accurate predictions.
* Assist farmers in early disease detection and prevention.

---

## 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib

---

## 📂 Dataset

**PlantVillage Dataset**

Supported Crops:

* Tomato
* Potato
* Bell Pepper

Total Classes: **15 Disease Categories**

---

## 🧠 CNN Architecture

* Input Layer (128×128×3 RGB Images)
* Conv2D + ReLU
* MaxPooling2D
* Conv2D + ReLU
* MaxPooling2D
* Flatten Layer
* Dense Layer (128 Neurons)
* Dropout Layer
* Softmax Output Layer

---

## ⚙ Features

* Plant Disease Classification
* Image Upload Interface
* Confidence Score Display
* Top-3 Predictions
* Disease Prevention Suggestions
* Interactive Streamlit Dashboard

---

## 📊 Model Performance

* Training Accuracy: ~90–92%
* Validation Accuracy: ~86–87%
* Test Accuracy: ~95%

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/yashaswini1220/Plant-Disease-Detection-CNN.git
```

### 2. Navigate to Project Folder

```bash
cd Plant-Disease-Detection-CNN
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
streamlit run app.py
```

---

## 📸 Application Workflow

1. Upload a leaf image.
2. Image is preprocessed and resized.
3. CNN model predicts the disease class.
4. Confidence score is generated.
5. Results and recommendations are displayed.

---

## 🔮 Future Enhancements

* Mobile Application Deployment
* Support for More Crop Types
* Real-Time Disease Detection
* Unknown Plant Detection
* Cloud-Based Deployment

---

## 👩‍💻 Author

**Yashaswini**

Plant Disease Detection Using CNN – Final Year Project
