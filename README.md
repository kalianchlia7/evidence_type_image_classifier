# Forensic-Image-Classifier
CNN classifier that categorizes real vs. AI-generated face images using TensorFlow in Python

# Description

This project is a binary image classifier that predicts whether a face image is real or AI-generated. The model uses a Convolutional Neural Network (CNN) in TensorFlow, including data preprocessing, optional augmentation, training, evaluation, and visualization. The trained model, performance plots, and confusion matrix are saved for reproducibility and analysis.

**Features**
- Loads and preprocesses real vs. fake face images
- Builds and trains a CNN using TensorFlow/Keras
- Evaluates model performance with accuracy, loss plots, and confusion matrix
- Saves trained model and training history for reuse without retraining
- Organized project structure for easy reproducibility

# Usage
```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("models/evidence_model.h5")

# Predict on new images
img_path = "data/test/real/sample_image.jpg"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction_prob = model.predict(img_array)
predicted_label = "Real" if prediction_prob > 0.5 else "Fake"
print(f"Predicted Label: {predicted_label}")

# Dataset

Dataset: Real vs. Fake Faces (AI-generated and real face images)
Structure: train/, val/, test/ folders with real/ and fake/ subfolders
Download link: (https://www.kaggle.com/datasets/ayushkvs/fake-and-real-face-detection)

Description:

#Accuracy Plot

accuracy_plot.png

#Loss Plot

loss_plot.png


