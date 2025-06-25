import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
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
    print("Loading dataset from local directory.")
    steel_data = pd.read_csv(file_path)
else:
    print("Dataset not found. Downloading from UCI repository.")
    dataset = fetch_ucirepo(id=198)
    steel_data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    os.makedirs("data", exist_ok=True)
    steel_data.to_csv(file_path, index=False)

# ==============================================================================
# 2. Quick Dataset Overview
# ==============================================================================
print("\n--- Data Summary ---")
print(steel_data.describe())
print(steel_data.head())
print(steel_data.info())

label_series = steel_data[steel_data.columns[-7:]].idxmax(axis=1)
plt.figure(figsize=(14, 4))
plt.plot(label_series.reset_index(drop=True), marker='.', linestyle='none')
plt.title('Fault Class Distribution by Row Index')
plt.ylabel('Class')
plt.xlabel('Index')
plt.show()

class_counts = steel_data[steel_data.columns[-7:]].sum()
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index, class_counts.values)
plt.title('Steel Fault Class Counts')
plt.xlabel('Fault Class')
plt.ylabel('Frequency')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             int(bar.get_height()), ha='center')
plt.tight_layout()
plt.show()

# ==============================================================================
# 3. Feature and Label Separation
# ==============================================================================
X = steel_data.iloc[:, :-7]
y_onehot = steel_data.iloc[:, -7:]
y_flat = np.argmax(y_onehot.values, axis=1)

# ==============================================================================
# 4. Feature Selection (Forward + Backward for KNN and NB)
# ==============================================================================
def score_knn(features):
    X_train, X_test, y_train, y_test = train_test_split(
        X[features], y_flat, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    return np.trace(cm) / cm.shape[0]

def score_nb(features):
    X_train, X_test, y_train, y_test = train_test_split(
        X[features], y_flat, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    y_pred = nb.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    return np.trace(cm) / cm.shape[0]

def forward_feature_selection(all_features, score_fn, max_features=5):
    selected, best_score = [], -1
    while len(selected) < max_features:
        candidates = []
        for f in all_features:
            if f not in selected:
                score = score_fn(selected + [f])
                candidates.append((f, score))
        best_feat, score = max(candidates, key=lambda x: x[1])
        if score > best_score:
            selected.append(best_feat)
            best_score = score
        else:
            break
    return selected

def backward_feature_elimination(initial_features, score_fn):
    features = list(initial_features)
    best_score = score_fn(features)
    while len(features) > 1:
        scores = []
        for f in features:
            temp = [feat for feat in features if feat != f]
            score = score_fn(temp)
            scores.append((f, score))
        worst_feat, score = min(scores, key=lambda x: x[1])
        if score < best_score:
            break
        features.remove(worst_feat)
        best_score = score
    return features

features = list(X.columns)

print("\n--- KNN Feature Selection ---")
knn_forward = forward_feature_selection(features, score_knn)
knn_backward = backward_feature_elimination(knn_forward, score_knn)
print("KNN Forward Selected:", knn_forward)
print("KNN Final Features (Backward Reduced):", knn_backward)

print("\n--- Naive Bayes Feature Selection ---")
nb_forward = forward_feature_selection(features, score_nb)
nb_backward = backward_feature_elimination(nb_forward, score_nb)
print("NB Forward Selected:", nb_forward)
print("NB Final Features (Backward Reduced):", nb_backward)

selected_features = knn_backward
X = steel_data[selected_features]

# ==============================================================================
# 5. Train/Test Split and Scaling
# ==============================================================================
X_train, X_test, y_train_flat, y_test_flat = train_test_split(
    X, y_flat, test_size=0.3, random_state=23
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_1hot = pd.get_dummies(y_train_flat)
y_test_1hot = pd.get_dummies(y_test_flat)

# ==============================================================================
# 6. KNN Hyperparameter Tuning
# ==============================================================================
print("\n--- KNN Hyperparameter Tuning ---")
best_k, best_acc = 0, 0
for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_1hot)
    y_pred = knn.predict(X_test_scaled)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test_flat, y_pred_labels, normalize='true')
    acc = np.trace(cm) / cm.shape[0]
    print(f"K = {k}: Accuracy = {acc:.4f}")
    if acc > best_acc:
        best_k, best_acc = k, acc

print(f"Best K: {best_k}")

# ==============================================================================
# 7. Evaluation Functions
# ==============================================================================
def evaluate_knn(knn_model, X_train, X_test, y_train, y_test):
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    y_true = np.argmax(y_test.values, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred_labels, normalize='true')
    acc = np.trace(cm) / cm.shape[0]
    print(f"KNN Accuracy: {acc:.2f}")
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title("KNN Confusion Matrix")
    plt.show()
    return acc

def evaluate_nb(nb_model, X_train, X_test, y_train, y_test):
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    acc = np.trace(cm) / cm.shape[0]
    print(f"Naive Bayes Accuracy: {acc:.2f}")
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Oranges')
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()
    return acc

# ==============================================================================
# 8. Final Model Comparison
# ==============================================================================
print("\n--- Final Model Comparison ---")
acc_knn = evaluate_knn(
    KNeighborsClassifier(n_neighbors=best_k),
    X_train_scaled, X_test_scaled,
    y_train_1hot, y_test_1hot
)

acc_nb = evaluate_nb(
    GaussianNB(),
    X_train_scaled, X_test_scaled,
    y_train_flat, y_test_flat
)

# ==============================================================================
# 9. Accuracy Bar Plot
# ==============================================================================
plt.figure(figsize=(6, 4))
plt.bar(["KNN", "Naive Bayes"], [acc_knn, acc_nb], color=["skyblue", "orange"])
plt.ylim(0, 1)
plt.ylabel("Mean Class Accuracy")
plt.title("Model Accuracy Comparison")
plt.text(0, acc_knn + 0.02, f"{acc_knn:.2f}", ha='center')
plt.text(1, acc_nb + 0.02, f"{acc_nb:.2f}", ha='center')
plt.tight_layout()
plt.show()

# ==============================================================================
# 10. Pairwise Scatter Plots (Selected Features)
# ==============================================================================
feature_pairs = list(combinations(selected_features, 2))
plots_per_fig = 6

for i in range(0, len(feature_pairs), plots_per_fig):
    subset = feature_pairs[i:i + plots_per_fig]
    plt.figure(figsize=(15, 10))

    for j, (f1, f2) in enumerate(subset):
        plt.subplot(2, 3, j + 1)
        plt.scatter(steel_data[f1], steel_data[f2], alpha=0.5)
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f"{f1} vs {f2}")

    plt.suptitle(f"Scatter Plots: Feature Relationships ({i + 1} to {i + len(subset)})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==============================================================================
# 11. Boxplots for All Features (Outlier Visualization)
# ==============================================================================
all_features = list(steel_data.columns[:-7])  # Exclude one-hot class columns
num_features = len(all_features)

for i in range(0, num_features, plots_per_fig):
    subset = all_features[i:i + plots_per_fig]
    plt.figure(figsize=(14, 10))
    for j, feat in enumerate(subset):
        plt.subplot(3, 2, j + 1)
        plt.boxplot(steel_data[feat], vert=False, patch_artist=True)
        plt.title(f"Boxplot of '{feat}'")
        plt.xlabel("Value")
    plt.suptitle(f"Outlier Visualization (Features {i + 1} to {i + len(subset)})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==============================================================================
# 12. Boxplots by Class (Selected Features)
# ==============================================================================
plot_df = steel_data[selected_features].copy()
plot_df['Class'] = y_flat

for feat in selected_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Class', y=feat, data=plot_df, color='#5DADE2')
    plt.title(f"Boxplot of '{feat}' by Fault Class")
    plt.xlabel("Fault Class")
    plt.ylabel(feat)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

