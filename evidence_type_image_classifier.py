# Evidence-Type Image Classifier (CNN) Project

# ==========================
# 1. Project Setup
# ==========================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import step 2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import step 4
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#import step 5
import pickle
#import step 6
from sklearn.metrics import confusion_matrix
import seaborn as sns
#import step 7
from tensorflow.keras.models import load_model

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ==========================
# 2. Data Loading & Exploration
# ==========================

# file path

# get the folder where this script lives
train_dir = "/Users/kalianchlia/Downloads/Coding Projects/ML Projects/Project 2/data/"

# Training data generator (W/ augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,         # normalize pixels
    validation_split=0.2,   # reserve 20% for validation
    rotation_range=20,       # randomly rotate images ±20 degrees
    width_shift_range=0.1,   # shift horizontally by ±10%
    height_shift_range=0.1,  # shift vertically by ±10%
    shear_range=0.1,         # apply shearing
    zoom_range=0.1,          # zoom in/out by ±10%
    horizontal_flip=True,    # flip images horizontally
    fill_mode='nearest'      # fill missing pixels after transformation
)

# Validation data generator (no augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255,         # normalize pixels
    validation_split=0.2
)

# Create training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # resize all images
    batch_size=32,
    class_mode='binary',      # binary classification
    subset='training',        # use 80% for training
    shuffle=True
)

# Create validation generator
val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',      # binary classification
    subset='validation',      # use 20% for validation
    shuffle=True
)


# ==========================
# 3. Data Preprocessing
# ==========================

# Check class labels and number of images
print("Class indices:", train_generator.class_indices)
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", val_generator.samples)

# Visualize a few sample images
x_batch, y_batch = next(train_generator)  # get first batch
plt.figure(figsize=(10,5))
for i in range(8):
    plt.subplot(2,4,i+1)                # 2 rows, 4 columns, position i+1
    plt.imshow(x_batch[i])               # show the image
    plt.title("Real" if y_batch[i]==0 else "Fake")  # add title based on label
    plt.axis('off')                      # hide axes for clarity
plt.show()

#Sanity check
x_batch, y_batch = next(train_generator)
print("Image batch shape:", x_batch.shape)  # (32, 128, 128, 3)
print("Label batch shape:", y_batch.shape)  # (32,)
print("Labels:", y_batch)                   # [0,1,0,1,...]

# x_batch.shape → (batch_size, height, width, channels(RGB))


# ==========================
# 4. Model Creation
# ==========================

# Sequential: build the CNN layer by layer in order#
# Conv2D: convolutional layer; extracts features like edges or patterns from images.
# MaxPooling2D: reduces spatial dimensions; keeps the most important features, helps reduce overfitting.
# Flatten: converts 2D feature maps into 1D vector so it can be fed into a dense layer.
# Dense: fully connected layer; does the actual classification.
# Dropout: randomly ignores some neurons during training to prevent overfitting.

model = Sequential()    #initialize model/create empty CNN model

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))  # 32 filters, 3x3 kernel, ReLU activation, input shape 128x128 RGB
model.add(MaxPooling2D((2,2)))  # max pooling with 2x2 window to reduce spatial dimensions

model.add(Conv2D(64, (3,3), activation='relu'))  # 64 filters, 3x3 kernel, ReLU activation
model.add(MaxPooling2D((2,2)))  # reduce spatial size again

model.add(Conv2D(128, (3,3), activation='relu'))  # 128 filters to capture more complex features
model.add(MaxPooling2D((2,2)))  # final reduction in spatial dimensions

model.add(Flatten())  # convert 3D feature maps into 1D vector
model.add(Dense(128, activation='relu'))  # fully connected layer
model.add(Dropout(0.5))                   # randomly drop 50% of neurons during training
model.add(Dense(1, activation='sigmoid')) # output layer for binary classification

model.compile(  #compile model
    optimizer='adam',
    loss='binary_crossentropy', #since classify-- real or fake
    metrics=['accuracy']
)

model.summary()

# ==========================
# 5. Model Training
# ==========================

#training parameters
epochs = 10        # Number of times the model will go through the entire training dataset
batch_size = 32    # Number of images processed at a time before updating weights

#train model
history = model.fit(
    train_generator,         # training data generator
    steps_per_epoch=train_generator.samples // batch_size,  # how many batches per epoch
    validation_data=val_generator,                           # validation data generator
    validation_steps=val_generator.samples // batch_size,   # validation batches per epoch
    epochs=epochs                                         # total epochs
)

print(history.history.keys())

# Save history object for later plotting or analysis
with open('history.pkl', 'wb') as file:
    pickle.dump(history.history, file)


# ==========================
# 6. Model Evaluation
# ==========================

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    train_dir,           # same folder containing real/ and fake/
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    shuffle=False        # VERY IMPORTANT: keep test data in same order
)

test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

#accuracy plot
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#loss plot
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

#confusion-matrix
#to see where the model misclassifies images
predictions = model.predict(test_generator)     #predicting from the test images
predicted_labels = (predictions > 0.5).astype(int).flatten()

true_labels = test_generator.classes
#actual class for each test image in order

cm = confusion_matrix(true_labels, predicted_labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.show()

# ==========================
# 7. Model Saving
# ==========================

model.save('models/evidence_model.h5')  #keras format, saves entire model
# (to load model later) model = load_model('models/evidence_model.h5')

#save training history with this
#with open('models/history.pkl', 'rb') as file:
    #loaded_history = pickle.load(file)

#to load saved plots later from history.pkl
"""
# Accuracy plot
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.savefig('plots/accuracy_plot.png')
plt.close()  # close figure to avoid overlapping plots

# Loss plot
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig('plots/loss_plot.png')
plt.close()
"""


# ==========================
# 8. Project Documentation
# ==========================
# - Structure GitHub repo
# - Include README with instructions and results