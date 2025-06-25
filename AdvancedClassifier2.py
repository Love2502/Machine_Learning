import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
# 1. Load Dataset
# ==============================================================================
file_path = os.path.join("data", "Faults.csv")

if os.path.exists(file_path):
    print("Loading dataset from local directory.")
    steel_data = pd.read_csv(file_path)
else:
    print("Downloading dataset from UCI repository.")
    dataset = fetch_ucirepo(id=198)
    steel_data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    os.makedirs("data", exist_ok=True)
    steel_data.to_csv(file_path, index=False)

# ==============================================================================
# 2. Prepare Features and Labels
# ==============================================================================
X_full = steel_data.iloc[:, :-7]
y_onehot = steel_data.iloc[:, -7:]
y_flat = np.argmax(y_onehot.values, axis=1)

# ==============================================================================
# 3. Combined Forward + Backward Feature Selection for SVM
# ==============================================================================
def svm_train_and_test(features):
    X_train, X_test, y_train, y_test = train_test_split(
        X_full[features], y_flat, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', gamma='scale')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    acc = np.trace(cm) / cm.shape[0]
    print(f"SVM Features: {features} Accuracy: {acc:.4f}")
    return acc

def combined_feature_selection_svm(all_feats, max_feats):
    selected = []
    while len(selected) < max_feats:
        best_feat, best_score = None, -1
        for f in all_feats:
            if f not in selected:
                score = svm_train_and_test(selected + [f])
                if score > best_score:
                    best_feat, best_score = f, score
        if best_feat:
            selected.append(best_feat)
        else:
            break

    # Backward elimination
    improved = True
    while improved and len(selected) > 1:
        current_score = svm_train_and_test(selected)
        scores = []
        for f in selected:
            reduced = selected.copy()
            reduced.remove(f)
            score = svm_train_and_test(reduced)
            scores.append((f, score))
        best_to_remove, best_score = max(scores, key=lambda x: x[1])
        if best_score > current_score:
            selected.remove(best_to_remove)
        else:
            improved = False

    return selected

print("\n--- SVM Feature Selection ---") #anyhow it will take time better to see what is happening in terminal
all_features = list(X_full.columns)
svm_selected_features = combined_feature_selection_svm(all_features, max_feats=5)
print("Selected features for SVM:", svm_selected_features)

# ==============================================================================
# 4. SVM Hyperparameter Tuning and Final Evaluation
# ==============================================================================
X = X_full[svm_selected_features]
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X, y_flat, test_size=0.3, random_state=42
)

scaler_svm = StandardScaler()
X_train_svm_scaled = scaler_svm.fit_transform(X_train_svm)
X_test_svm_scaled = scaler_svm.transform(X_test_svm)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

print("\n--- Performing Grid Search for SVM ---")
grid_search = GridSearchCV(
    SVC(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_svm_scaled, y_train_svm)
best_svm = grid_search.best_estimator_

print("\nBest Parameters Found:")
print(grid_search.best_params_)

# ==============================================================================
# 5. Summary Table and Line Plot of Grid Results
# ==============================================================================
results_df = pd.DataFrame(grid_search.cv_results_)

print("\nTop 5 Hyperparameter Combinations:")
print(results_df[['mean_test_score', 'params']].sort_values(by='mean_test_score', ascending=False).head())

plt.figure(figsize=(8, 5))
for kernel in results_df['param_kernel'].unique():
    subset = results_df[
        (results_df['param_kernel'] == kernel) &
        ((results_df['param_gamma'] == 'scale') | (results_df['param_gamma'] == 0.1))
    ]
    plt.plot(subset['param_C'], subset['mean_test_score'], marker='o', label=f'kernel={kernel}')

plt.title('SVM Accuracy vs. C (for gamma=scale or 0.1)')
plt.xlabel('C Value')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================================================================
# 6. Evaluate Best SVM on Test Set
# ==============================================================================
y_pred_svm = best_svm.predict(X_test_svm_scaled)
cm_svm = confusion_matrix(y_test_svm, y_pred_svm, normalize='true')
acc_svm = np.trace(cm_svm) / cm_svm.shape[0]
print(f"Tuned SVM Accuracy: {acc_svm * 100:.2f}%")

ConfusionMatrixDisplay(confusion_matrix=cm_svm).plot(cmap='Purples')
plt.title("Tuned SVM Confusion Matrix")
plt.show()
