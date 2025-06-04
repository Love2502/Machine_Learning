import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
# 1. Load Dataset (from file if exists, else download from UCI)
# ==============================================================================
file_path = os.path.join("data", "Faults.csv")

if os.path.exists(file_path):
    # If we have the CSV locally, read it directly for speed
    print("Loading dataset from local directory.")
    steel_data = pd.read_csv(file_path)
else:
    # Otherwise, fetch from UCI repository (id=198 for Steel Plates Faults)
    print("Dataset not found. Downloading from UCI repository.")
    dataset = fetch_ucirepo(id=198)
    # Concatenate the feature columns and the one-hot encoded target columns
    steel_data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    os.makedirs("data", exist_ok=True)
    steel_data.to_csv(file_path, index=False)
    # We save locally so next time we don't re-download

# ==============================================================================
# 2. Quick Dataset Overview
# ==============================================================================
print("\n--- Data Summary ---")
print(steel_data.describe())  # Basic statistics on each numeric feature
print(steel_data.head())      # Show first five rows
print(steel_data.info())      # Data types and non-null counts

# The target columns are one-hot encoded (7 possible fault types).  To
# visualize how faults are distributed across row indices:
label_series = steel_data[steel_data.columns[-7:]].idxmax(axis=1)
# idxmax finds which of the 7 fault columns is “1” for each row
plt.figure(figsize=(14, 4))
plt.plot(label_series.reset_index(drop=True), marker='.', linestyle='none')
plt.title('Fault Class Distribution by Row Index')
plt.ylabel('Class')
plt.xlabel('Index')
plt.show()

# Next, a bar chart of the raw counts of each fault:
class_counts = steel_data[steel_data.columns[-7:]].sum()
# Summing each one-hot column gives frequency of that fault
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index, class_counts.values)
plt.title('Steel Fault Class Counts')
plt.xlabel('Fault Class')
plt.ylabel('Frequency')
for bar in bars:
    # Annotate counts above each bar
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 5,
             int(bar.get_height()),
             ha='center')
plt.tight_layout()
plt.show()

# ==============================================================================
# 3. Feature and Label Separation
# ==============================================================================
# The last 7 columns are one-hot encoded target variables (7 fault types).
X = steel_data.iloc[:, :-7]          # All columns except the last 7 → features
y_onehot = steel_data.iloc[:, -7:]   # Last 7 columns ,one-hot fault labels
# For Naive Bayes, we need a flat integer label.  np.argmax selects the index
# of “1” in each row, resulting in 0–6 integer labels.
y_flat = np.argmax(y_onehot.values, axis=1)

# ==============================================================================
# 4. Feature Selection with KNN
# ==============================================================================
def train_and_test(features):
    """
    Given a set of feature names, train a KNN model and report mean class-wise accuracy.
    We split into 70% train, 30% test each time, scale, then fit KNN with k=5.
    """
    # 1. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X[features], y_flat, test_size=0.3, random_state=23
    )

    # 2. Scale features => KNN uses Euclidean distance.  Without scaling,
    #    features with larger numeric range dominate the distance metric.
    scaler = StandardScaler()
    # Fit scaler on training data (compute mean μ and std σ), then transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Fit KNN with k=5 (a common default to balance bias/variance)
    knn = KNeighborsClassifier(n_neighbors=5)
    # scikit‐learn’s KNeighborsClassifier expects either numeric labels or one-hot
    # encoding for multi‐class.  Here we provide one-hot so that predict() returns
    # a one-hot vector for each instance; argmax below recovers label index.
    knn.fit(X_train_scaled, pd.get_dummies(y_train))

    # 4. Predict on test set
    y_pred = knn.predict(X_test_scaled)
    # Convert one-hot back to flat integer labels
    y_pred_labels = np.argmax(y_pred, axis=1)

    # 5. Build confusion matrix (rows=true class, cols=predicted class)
    cm = confusion_matrix(y_test, y_pred_labels, normalize='true')
    # The “normalize='true'” option scales each row to sum to 1 => per-class recall
    # To get mean class‐wise accuracy, sum diagonal (correct rate per class),
    # then divide by number of classes (=7).
    acc = np.trace(cm) / cm.shape[0]
    
    # Format the output with fixed width for alignment (Not important)
    features_str = str(features).ljust(150)  
    print(f"Features: {features_str} Accuracy: {acc:>7.4f}") 
    return acc

def forward_feature_selection(all_feats, max_feats):
    """
    Greedy forward selection:
      - Start with empty selected set, best_score = -1
      - Iteratively add the feature that yields highest accuracy when combined
        with already-selected features.
      - Stop if no feature can improve accuracy further, or if max_feats reached.
    """
    selected, best_score = [], -1
    while len(selected) < max_feats:
        candidates = []
        # Try adding each remaining feature to current selected set:
        for f in all_feats:
            if f not in selected:
                score = train_and_test(selected + [f])
                candidates.append((f, score))
        # Pick feature with highest score
        best_feat, score = max(candidates, key=lambda x: x[1])
        if score > best_score:
            selected.append(best_feat)
            best_score = score
        else:
            # No improvement => stop
            break
    return selected

print("\n--- Feature Selection ---")
features = list(X.columns)                          # All column names as candidate features
selected_features = forward_feature_selection(features, max_feats=5)
print("Selected features:", selected_features)

# Keep only the selected features in X
X = steel_data[selected_features]


# =============================================================================
#                            Data Cleaning
# =============================================================================

# print("\n================= Removing the Outlier =================")

# df_features_selected = pd.concat([X,y_onehot], axis=1)

# # Remove outliers
# def outliars(data):
#     for col in data.select_dtypes(include='number').columns:
#         q1 = data[col].quantile(0.25)
#         q3 = data[col].quantile(0.75)
#         iqr = q3 - q1
#         lower = q1 - 1.5 * iqr
#         upper = q3 + 1.5 * iqr
#         data = data[(data[col] >= lower) & (data[col] <= upper)]
#     return data

# df_cleaned = outliars(df_features_selected)

# cleaned_classes = df_cleaned[df_cleaned.columns[-7:]]


# =============================================================================
#                   Data Visualization of the cleaned data
# =============================================================================

# print("\n================= Visualizing the Data after cleaning =================")

# class_counts = cleaned_classes.sum()

# # Bar plot
# plt.figure(figsize=(10, 6))
# bars = plt.bar(class_counts.index, class_counts.values)

# plt.title('Steel Fault Class Distribution after Outlier removal')
# plt.xlabel('Fault Class')
# plt.ylabel('Number of Instances')

# # Add numbers on top of bars
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), 
#              ha='center', va='bottom', fontsize=10)

# plt.tight_layout()
# plt.show()


# # This code was used to generate reports on the data after outliers were removed
# '''
# from ydata_profiling import ProfileReport

# profile = ProfileReport(df_features_selected, title="df_features_selected Report")
# profile.to_file("df_features_selected_report.html")

# profile = ProfileReport(df_cleaned, title="df_cleaned Report")
# profile.to_file("df_cleaned_report.html")
# '''

# print(df_cleaned.head())
# print(df_cleaned.describe())
# print(df_cleaned.shape)

# X = df_cleaned.iloc[:, :-7]  
# y_onehot = df_cleaned.iloc[:, -2:]
# # For Naive Bayes, we need a flat integer label.  np.argmax selects the index
# # of “1” in each row, resulting in 0–6 integer labels.
# y_flat = np.argmax(y_onehot.values, axis=1)


# ==============================================================================
# 5. Train/Test Split and Scaling (final data)
# ==============================================================================
# Now that we know which features matter, split once.  We’ll reuse these splits
# for both KNN and Naive Bayes comparisons.
X_train, X_test, y_train_flat, y_test_flat = train_test_split(
    X, y_flat, test_size=0.3, random_state=23
)

# Scale train/test using statistics from training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN expects one-hot labels for predict() in multi-class setting:
y_train_1hot = pd.get_dummies(y_train_flat)
y_test_1hot = pd.get_dummies(y_test_flat)

# ==============================================================================
# 6. KNN Hyperparameter Tuning (search for best k)
# ==============================================================================
print("\n--- KNN Hyperparameter Tuning ---")
best_k, best_acc = 0, 0
for k in range(1, 10):
    # Build model with current k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_1hot)
    y_pred = knn.predict(X_test_scaled)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Build normalized confusion matrix => each row sums to 1 => per-class recall
    cm = confusion_matrix(y_test_flat, y_pred_labels, normalize='true')
    # We want the average of those 7 diagonal entries => mean class-wise accuracy
    acc = np.trace(cm) / cm.shape[0]
    # print(f"K = {k}: Accuracy = {acc:.4f}")
    print(f"Average of diagonal (mean class-wise accuracy) for KNN with k = {k}: {acc:.4f}")
    if acc > best_acc:
        best_k, best_acc = k, acc

# print(f"Best K = {best_k} with accuracy = {best_acc:.4f}")
print(f"After iterating through the values of K, the best ones is {best_k}")

# ==============================================================================
# 7. Evaluation Functions
# ==============================================================================
def evaluate_knn(knn_model, X_train, X_test, y_train, y_test):
    """
    Final KNN evaluation:
      - Fit on X_train/y_train (both one-hot)
      - Predict on X_test ⇒ get one-hot outputs
      - Compare argmax of predictions to true labels (argmax of y_test)
      - Compute confusion matrix and average of diagonal = class‐wise accuracy
    """
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    y_true = np.argmax(y_test.values, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Confusion matrix: rows=true classes, cols=predicted classes; normalize row‐wise
    cm = confusion_matrix(y_true, y_pred_labels, normalize='true')
    # Mean class‐wise accuracy = average of diagonal entries
    acc = np.trace(cm) / cm.shape[0]
    print(f"KNN Accuracy: {acc:.4f}")
    # Display confusion matrix for visual inspection
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title("KNN Confusion Matrix")
    plt.show()
    return acc

def evaluate_nb(nb_model, X_train, X_test, y_train, y_test):
    """
    Naive Bayes evaluation:
      - Fit GaussianNB on (X_train, y_train) where y_train is flat integer labels
      - Predict flat labels on X_test
      - Compute confusion matrix (flat labels vs. flat labels)
      - Average diagonal entries for mean class‐wise accuracy
    """
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    acc = np.trace(cm) / cm.shape[0]
    print(f"Naive Bayes Accuracy: {acc:.4f}")
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Oranges')
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()
    return acc

# Assuming 'selected_features' is a list of 5 feature names
# and 'steel_data' is your full DataFrame including these features

df_selected = steel_data[selected_features]

plt.figure(figsize=(12, 8))

for i, feature in enumerate(selected_features):
    plt.subplot(2, 3, i + 1)
    plt.scatter(df_selected.index, df_selected[feature], color='teal', alpha=0.6)
    plt.title(feature)
    plt.xlabel("Index")
    plt.ylabel("Value")

plt.tight_layout()
plt.show()

# ==============================================================================
# 8. Final Model Comparison (with best K from above)
# ==============================================================================
print("\n--- Final Model Comparison ---")
# KNN: supply best_k, trained on one-hot y_train_1hot
acc_knn = evaluate_knn(
    KNeighborsClassifier(n_neighbors=best_k),
    X_train_scaled, X_test_scaled,
    y_train_1hot, y_test_1hot
)
# Naive Bayes: train on flat integer labels
acc_nb = evaluate_nb(
    GaussianNB(),
    X_train_scaled, X_test_scaled,
    y_train_flat, y_test_flat
)

# ==============================================================================
# 9. Accuracy Comparison Plot
# ==============================================================================
plt.figure(figsize=(6, 4))
plt.bar(["KNN", "Naive Bayes"], [acc_knn, acc_nb], color=["skyblue", "orange"])
plt.ylim(0, 1)
plt.ylabel("Mean Class Accuracy")
plt.title("Model Accuracy Comparison")
# Annotate the bars with their numeric scores
plt.text(0, acc_knn + 0.02, f"{acc_knn:.4f}", ha='center')
plt.text(1, acc_nb + 0.02, f"{acc_nb:.4f}", ha='center')
plt.tight_layout()
plt.show()
