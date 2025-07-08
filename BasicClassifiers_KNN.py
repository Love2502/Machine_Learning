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
# Setup Output Directory for Plots
# ==============================================================================
os.makedirs("plots", exist_ok=True)

# ==============================================================================
# 1. Load Dataset (from file if exists, else download from UCI)
# ==============================================================================
file_path = os.path.join("data/data", "Faults.csv")

if os.path.exists(file_path):
    print("Loading dataset from local directory.")
    steel_data = pd.read_csv(file_path)
else:
    print("Dataset not found. Downloading from UCI repository.")
    dataset = fetch_ucirepo(id=198)
    steel_data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    os.makedirs("data/data", exist_ok=True)
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
# plt.savefig("plots/fault_class_distribution.svg", format="svg")
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
# plt.savefig("plots/fault_class_counts.svg", format="svg")
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



# def score_knn(features):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X[features], y_flat, test_size=0.3, random_state=94469
#     )
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(X_train_scaled, y_train)
#     y_pred = knn.predict(X_test_scaled)
#     cm = confusion_matrix(y_test, y_pred, normalize='true')
#     return np.trace(cm) / cm.shape[0]


# def feature_selection(all_features, score_fn, max_features=5):
#     selected = []
#     best_score = float("-inf")
#     print("=== feature selection ===\n")

#     while len(selected) < max_features:
#         # ---- Forward step ----
#         print(f"Forward step (selected={selected}, best_score={best_score:.4f})")
#         fwd_candidates = []
#         for f in all_features:
#             if f not in selected:
#                 s = score_fn(selected + [f])
#                 fwd_candidates.append((f, s))
#                 print(f"  Trying to add '{f}': score = {s:.4f}")
#         best_f, best_f_score = max(fwd_candidates, key=lambda x: x[1])

#         if best_f_score > best_score:
#             selected.append(best_f)
#             best_score = best_f_score
#             print(f"  -> Added '{best_f}', new best_score = {best_score:.4f}\n")
#         else:
#             print("  -> No addition improves; stopping.\n")
#             break

#         # ---- Backward step(s) ----
#         print("  Backward step:")
#         while len(selected) > 1:
#             bwd_candidates = []
#             for f in selected:
#                 subset = [feat for feat in selected if feat != f]
#                 s = score_fn(subset)
#                 bwd_candidates.append((f, s))
#                 print(f"    Trying to remove '{f}': score = {s:.4f}")
#             # pick the removal that gives highest score
#             worst_f, worst_f_score = max(bwd_candidates, key=lambda x: x[1])

#             if worst_f_score > best_score:
#                 selected.remove(worst_f)
#                 best_score = worst_f_score
#                 print(f"    -> Removed '{worst_f}', new best_score = {best_score:.4f}\n")
#             else:
#                 print("    -> No removal improves; end backward step.\n")
#                 break

#     print(f"Final selected features ({len(selected)}): {selected}\n")
#     return selected


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def evaluate_features_with_k(features, k, cv_splits=5):
    """
    Perform cross-validation with standardization and return average classwise classification rate.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    avg_diagonal = []

    for train_idx, test_idx in skf.split(X[features], y_flat):
        X_train, X_test = X.iloc[train_idx][features], X.iloc[test_idx][features]
        y_train, y_test = y_flat[train_idx], y_flat[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred, normalize='true')
        avg_diagonal.append(np.trace(cm) / cm.shape[0])  # average per-class acc

    return np.mean(avg_diagonal)


def forward_feature_selection_k_optimization(all_features, max_features=10):
    """
    For each k from 1 to 24, perform forward selection and keep best k and features.
    """
    best_global_score = -np.inf
    best_global_k = None
    best_global_features = []

    for k in [x for x in range(1, 25) if x % 2 != 0]:
        selected = []
        current_best_score = -np.inf
        print(f"\n=== Forward Feature Selection for K = {k} ===")

        while len(selected) < max_features:
            candidates = []
            for feat in all_features:
                if feat not in selected:
                    test_features = selected + [feat]
                    score = evaluate_features_with_k(test_features, k)
                    candidates.append((feat, score))
                    print(f"  Testing '{feat}' -> score: {score:.4f}")

            best_feat, best_score = max(candidates, key=lambda x: x[1])

            if best_score > current_best_score:
                selected.append(best_feat)
                current_best_score = best_score
                print(f"  -> Selected '{best_feat}', score = {best_score:.4f}")
            else:
                break

        print(f"Finished selection for k = {k}. Score = {current_best_score:.4f} | Features: {selected}")

        if current_best_score > best_global_score:
            best_global_score = current_best_score
            best_global_k = k
            best_global_features = selected.copy()

    print(f"\n\n=== Best Overall ===")
    print(f"Best K: {best_global_k}")
    print(f"Best Score: {best_global_score:.4f}")
    print(f"Best Features: {best_global_features}")
    return best_global_k, best_global_features


# # --- Example usage with your KNN scorer ---
# features = list(X.columns)
# knn_final = feature_selection(features, score_knn, max_features=10)

# knn_backward = knn_final

# selected_features = knn_backward
# X = steel_data[selected_features]


features = list(X.columns)
best_k, best_features = forward_feature_selection_k_optimization(features, max_features=10)
X = steel_data[best_features]



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
for k in range(1, 24):
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
import matplotlib.patches as patches

def evaluate_knn(knn_model, X_train, X_test, y_train, y_test):
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    y_true = np.argmax(y_test.values, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred_labels, normalize='true')
    acc = np.trace(cm) / cm.shape[0]
    print(f"KNN Accuracy: {acc:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax,
        linewidths=0,
        annot_kws={"fontsize": 10}
    )

    for text in ax.texts:
        if text.get_text() == '0.000':
            text.set_text('0')

    border = patches.Rectangle((0, 0), cm.shape[0], cm.shape[1], fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(border)

    plt.title("KNN Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    # plt.savefig("plots/knn_confusion_matrix.svg", format="svg")
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
# ==============================================================================
# 9. Accuracy Bar Plot
# ==============================================================================
plt.figure(figsize=(6, 4))
plt.bar(["KNN"], [acc_knn], color=["skyblue"])
plt.ylim(0, 1)
plt.ylabel("Mean Class Accuracy")
plt.text(0, acc_knn + 0.02, f"{acc_knn:.2f}", ha='center')
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
    # plt.savefig(f"plots/scatter_plots_{i+1}_to_{i+len(subset)}.svg", format="svg")
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
    # plt.savefig(f"plots/boxplot_by_class_{feat}.svg", format="svg")
    plt.show()

