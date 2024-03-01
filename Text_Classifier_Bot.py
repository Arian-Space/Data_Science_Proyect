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

# Read data from a TXT file
with open('link_txt', 'r') as file:
  lines = file.readlines()

# Split lines in text and labels
sentences = []
labelss = []
for line in lines:
  parts = line.strip().split()
  text = ' '.join(parts[1:])
  label = str(parts[0])
  sentences.append(text)
  labelss.append(label)

# # Tokenization and text sequencing

max_words = 1000  # Maximum number of words in the vocabulary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# # Padding sequences so they are the same length

max_sequence_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# New arrangement
labels = []

# Go to binary
for i in labelss:
  if i == 'ham':
    labels.append(0)
  else:
    labels.append(1)

# Transform arrays in numpy
sentences = np.array(sentences)
labels = np.array(labels)

# # Checking the variables

# print(sentences)
# print(len(labels))

# # Model creation function

def design_model(training_data):
  print("\nBuilding model...")

  # Create the text classification model
  model = tf.keras.Sequential(name='LSTM_spam_AI')
  model.add(Embedding(max_words, 64, input_length=max_sequence_length))
  model.add(Bidirectional(LSTM(64)))
  model.add(Dense(1, activation='sigmoid'))

  # Compile the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # Summarize model
  model.summary()

  return model

# Use model function
model = design_model(sentences)

# Train the model
history = model.fit(sequences, labels, epochs=6, validation_split=0.2)

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

# # Prediction function

# Read data from a TXT file
with open('link_2', 'r') as file:
    lines = file.readlines()

# Split lines in text and labels
sentences_val = []
labelss_val = []
for line in lines:
    parts = line.strip().split()
    text = ' '.join(parts[1:])
    label = str(parts[0])
    sentences_val.append(text)
    labelss_val.append(label)

# Re-definition
'''
sentences_val = sentences_val[0:100]
labelss_val = labelss_val[0:100]
'''

# Tokenization and text sequencing
max_words = 1000  # Maximum number of words in the vocabulary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(sentences_val)
sequences = tokenizer.texts_to_sequences(sentences_val)

'''
# Padding sequences so they are the same length
max_sequence_length = max(len(seq) for seq in sentences_val)
sentences_val = pad_sequences(sequences, maxlen=max_sequence_length)
'''

# New arrangement validation
labels_val = []

# Go to binary
for i in labelss_val:
  if i == 'ham':
    labels_val.append(0)
  else:
    labels_val.append(1)

# Transform arrays in numpy
sentences_val = np.array(sentences_val)
labels_val = np.array(labels_val)

# # Checking the variables

# print(len(sentences_val))
# print(len(labels_val))

# # Classify new text

labels_val
# Predict results
new_sequence = tokenizer.texts_to_sequences(sentences_val)
new_sequence = pad_sequences(new_sequence, maxlen=171)
prediction = model.predict(new_sequence)

predictionBin = []

# Examine probability
for i in prediction:
  if i < 0.5:
    predictionBin.append(0)
  else:
    predictionBin.append(1)

# print("Prediction:", predictionBin)

# Examine validation
resultFin = []
for i in range(len(labels_val)):
 if labels_val[0] == predictionBin[0]:
  resultFin.append(labels_val[0])

# Final score
print('The probability for a good clasification is:', round(len(resultFin)/len(labels_val)*100,2))
