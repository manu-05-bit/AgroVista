import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Load the dataset
data = pd.read_csv('Crop.csv')
print(data)

# Encode the target labels (crop names)
label_encoder = LabelEncoder()
encoded_crops = label_encoder.fit_transform(data.iloc[:, 7])
print(encoded_crops)

# Select features (X) and target (y)
X = data.iloc[:, 0:7]
y = encoded_crops

# Split the data into training, validation, and test sets
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.2, random_state=2022)

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# Initialize and train the Random Forest model
RF = RandomForestClassifier()
RF.fit(X_train, y_train)

# Calculate accuracies
training_accuracy = metrics.accuracy_score(y_train1, RF.predict(X_train1)) * 100
validation_accuracy = metrics.accuracy_score(y_val, RF.predict(X_val)) * 100
test_accuracy = RF.score(X_test, y_test)

# Print the accuracies
print(f'Training Accuracy: {training_accuracy:.2f}%')
print(f'Validation Accuracy: {validation_accuracy:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
