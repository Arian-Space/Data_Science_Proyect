# # Libraries to use

# Data extraction
import io
import os
import requests
import numpy as np

# Display function
from matplotlib import pyplot as plt

# For machine learning
# Using pip for instaling => !pip install tensorflow==2.8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import numpy

# For image
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
import matplotlib.pyplot as plt

# # Variables and data

# Link containing training data
LINK_TRAIN = "link_1"
# Link containing validation data
LINK_VAL = "link_2"
# Type of data to take into account
CLASS_MODE = "binary"
# Type of color to take into account (grayscale)
COLOR_MODE1 = "grayscale"
# Type of color to take into account (rgb)
COLOR_MODE2 = "rgb"
# Size in pixels
TARGET_SIZE = (150, 150)
# Batch Size
BATCH_SIZE = 3

# Object creator (training)
train_datagen = ImageDataGenerator(rescale=1.0/255,zoom_range=0.1,rotation_range=25,width_shift_range=0.05,height_shift_range=0.05)
# Object creator (validation)
validation_datagen = ImageDataGenerator(rescale=1.0/255,zoom_range=0.1,rotation_range=25,width_shift_range=0.05,height_shift_range=0.05)

# Validation data generator
train_generator = train_datagen.flow_from_directory(
    LINK_TRAIN,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    color_mode=COLOR_MODE2
)

# Validation data generator
validation_generator = validation_datagen.flow_from_directory(
    LINK_VAL,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    color_mode=COLOR_MODE2
)
  
# # Function to build the model

def design_model(training_data):
  print("\nBuilding model...")

  # Define the CNN model
  model = Sequential(name='CyD_AI')
  # Convolutional network input neurons
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
  # Experimental pooling
  # model.add(layers.MaxPooling2D((2, 2)))
  # The maxpooling layers and dropout layers as well
  model.add(layers.Conv2D(64, (3, 3), strides=3, activation="relu"))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  model.add(layers.Dropout(0.1))
  model.add(layers.Conv2D(128, (3, 3), strides=1, activation="relu"))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  model.add(layers.Dropout(0.2))

  # Experimenting with extra layer's
  model.add(tf.keras.layers.Conv2D(3, 3, strides=1, activation="relu"))
  model.add(tf.keras.layers.Conv2D(1, 1, strides=1, activation="relu"))
  model.add(tf.keras.layers.Dropout(0.1))

  model.add(layers.Flatten())
  # Hidden layer (activation fun = reLU)
  model.add(layers.Dense(32, activation = "relu"))
  # Output layer with softmax activation function
  model.add(layers.Dense(1,activation="sigmoid")) # sigmoid
  # Compile model with Adam optimizer
  # Loss function is categorical crossentropy
  # Compiling the CNN model
  print("\nCompiling model...")
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

  # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()],)

  # Summarize model
  model.summary()

  print('Model done.')
  return model

# Use model function
model = design_model(train_generator)
  
# # Training the model

history = model.fit(
    train_generator,
    steps_per_epoch=200,  # Number of steps per epoch
    epochs=20,  # Number of training epochs
    validation_data=validation_generator, # Our validation for training data
    validation_steps=100 # Number of validation steps
)

# # Plot using the Matplotlib library

# Graph training metrics
acc = history.history['accuracy'] # Accuracy
val_acc = history.history['val_accuracy'] # Accuracy validation
loss = history.history['loss'] # Loss from accuracy
val_loss = history.history['val_loss'] # Loss from accuracy validation

epochs = range(1, len(acc) + 1) # Epochs for data

plt.figure(figsize=(12, 4)) # Graph size

plt.subplot(1, 2, 1) # The accuracy graph
plt.plot(epochs, acc, color='red', label='Accuracy') # Taking the data
plt.plot(epochs, val_acc, label='Validation accuracy') # Taking the data for validation
plt.title('Accuracy of training and validation') # Title
plt.xlabel('Epochs') # X-label Epochs
plt.ylabel('Accuracy') # Y-label Accuracy
plt.legend() # Legend for data

plt.subplot(1, 2, 2) # The loss graph
plt.plot(epochs, loss, color='red', label='Loss graph') # Taking the data
plt.plot(epochs, val_loss, label='Loss of validation') # Taking the data for validation
plt.title('Loss of training and validation') # Title
plt.xlabel('Epochs') # X-label Epochs
plt.ylabel('Accuracy') # Y-label Accuracy
plt.legend() # Legend for data

plt.show() # Show graph
