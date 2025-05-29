import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo

file_path = "data/Faults.csv"

# Checks if dataset file exists
if os.path.exists(file_path):
    print("Loading dataset into memory")
    steel_data = pd.read_csv(file_path)
else:
    print("Dataset not found in /data")
    print("Downloading dataset from https://archive.ics.uci.edu/dataset/198/steel+plates+faults")
    # Fetch dataset from UCI
    # Source of this code: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
    steel_plates_faults = fetch_ucirepo(id=198)
    
    # Extract features and targets
    X = steel_plates_faults.data.features
    y = steel_plates_faults.data.targets

    # Combine into one DataFrame
    steel_data = pd.concat([X, y], axis=1)

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    steel_data.to_csv(file_path, index=False)

# Continue with analysis
print(steel_data.describe())
print(steel_data.head())
print(steel_data.info())

# Choose one fault type to classify — binary classification (0 or 1)
target_fault = 'Pastry'

# Split features and labels
X = steel_data.iloc[:, :-7].values  # All columns except last 7 — input features
y = steel_data[target_fault].values  # The column we are trying to predict

# Manual train/test split: 80% training, 20% testing
split_index = int(0.8 * len(X))
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# KNN function
def knn_predict(X_train, y_train, x_test, num_neighbors=3):
    distances = []  # To store distances to all training points

    # Calculate Euclidean distance from x_test to each x_train
    for idx in range(len(X_train)):
        distance = np.sqrt(np.sum((x_test - X_train[idx]) ** 2))
        distances.append((distance, y_train[idx]))  # Store (distance, label)

    # Sort the list of distances in ascending order
    distances.sort(key=lambda tup: tup[0])

    # Get the labels of the k nearest neighbors
    k_labels = [label for _, label in distances[:num_neighbors]]

    # Count the number of 0s and 1s in the nearest neighbors
    count_0 = np.sum(np.array(k_labels) == 0)
    count_1 = np.sum(np.array(k_labels) == 1)

    # Return the label that occurs more often
    return 1 if count_1 > count_0 else 0

# Run prediction for each test point
k_neighbors = 5  # Number of neighbors
predictions = []  # To store all predicted values
print(predictions)

for test_idx in range(len(X_test)):
    pred = knn_predict(X_train, y_train, X_test[test_idx], num_neighbors=k_neighbors)
    predictions.append(pred)

# Evaluate accuracy: how many predictions were correct
correct = sum(pred == true for pred, true in zip(predictions, y_test))
accuracy = correct / len(y_test)
print(f"KNN Accuracy (k={k_neighbors}) for fault type '{target_fault}': {accuracy:.2f}")

# 1. Visualization of Actual vs Predicted values
#plt.figure(figsize=(10, 6))
#plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
#plt.scatter(range(len(predictions)), predictions, color='red', alpha=0.5, label='Predicted')
#plt.title('Actual vs Predicted values')
#plt.xlabel('Test Data Points')
#plt.ylabel('Fault Type')
#plt.legend()
#plt.show()

# Confusion Matrix
# Confusion matrix is a 2x2 matrix for binary classification
tp = np.sum((y_test == 1) & (np.array(predictions) == 1))  # True positives
fp = np.sum((y_test == 0) & (np.array(predictions) == 1))  # False positives
fn = np.sum((y_test == 1) & (np.array(predictions) == 0))  # False negatives
tn = np.sum((y_test == 0) & (np.array(predictions) == 0))  # True negatives

# Plot confusion matrix
cm = np.array([[tp, fp], [fn, tn]])
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')  # Fixed colormap name usage
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['No Fault', 'Fault']
tick_marks = np.arange(len(classes))
#plt.xticks(tick_marks, classes)
#plt.yticks(tick_marks, classes)

#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()
