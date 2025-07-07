import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
###############      RANDOM FOREST      ###############
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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
# 2. Prepare Features and Target
# ==============================================================================
X_full = steel_data.iloc[:, :-7]
y_onehot = steel_data.iloc[:, -7:]
y_flat = np.argmax(y_onehot.values, axis=1)

# ==============================================================================
# 3. Combined Forward and Backward Feature Selection
# ==============================================================================
def rf_train_and_test(features):
    X_train, X_test, y_train, y_test = train_test_split(
        X_full[features], y_flat, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    acc = np.trace(cm) / cm.shape[0]
    print(f"RF Features: {features} Accuracy: {acc:.4f}")
    return acc

def combined_feature_selection(all_feats, max_feats):
    selected = []
    while len(selected) < max_feats:
        best_feat, best_score = None, -1
        for f in all_feats:
            if f not in selected:
                score = rf_train_and_test(selected + [f])
                if score > best_score:
                    best_feat, best_score = f, score
        if best_feat:
            selected.append(best_feat)
        else:
            break

    # Backward step: remove features that reduce performance
    improved = True
    while improved and len(selected) > 1:
        current_score = rf_train_and_test(selected)
        scores = []
        for f in selected:
            reduced = selected.copy()
            reduced.remove(f)
            score = rf_train_and_test(reduced)
            scores.append((f, score))
        best_to_remove, best_score = max(scores, key=lambda x: x[1])
        if best_score > current_score:
            selected.remove(best_to_remove)
        else:
            improved = False

    return selected

print("\n--- Combined Forward and Backward Feature Selection ---")
all_features = list(X_full.columns)
selected_rf_features = combined_feature_selection(all_features, max_feats=5)
print("Selected features for Random Forest:", selected_rf_features)

# ==============================================================================
# 4. Train/Test Split and Scaling
# ==============================================================================
X = X_full[selected_rf_features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_flat, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 5. Hyperparameter Tuning with Grid Search
# ==============================================================================
print("\n--- Performing Grid Search for Random Forest ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': [None, 'sqrt']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_
print("\nBest Parameters Found:")
print(grid_search.best_params_)

# ==============================================================================
# 6. Final Evaluation with Tuned Model
# ==============================================================================
print("\n--- Evaluating Tuned Random Forest ---")
y_pred_rf = best_rf.predict(X_test_scaled)
cm_rf = confusion_matrix(y_test, y_pred_rf, normalize='true')
acc_rf = np.trace(cm_rf) / cm_rf.shape[0]
print(f"Random Forest Accuracy: {acc_rf:.4f}")

ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot(cmap='Greens')
plt.title("Tuned Random Forest Confusion Matrix")
plt.show()
