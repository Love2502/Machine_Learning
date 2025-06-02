import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# Start timer
print("Start Time:", time.process_time())

# ✅ Fast forward selection
def forward_selection_fast(X, y, max_features, feature_names, force_full=True):
    selected = []
    best_score = -np.inf
    model = GaussianNB()

    while len(selected) < max_features:
        scores = []
        for i in range(X.shape[1]):
            if i not in selected:
                temp = selected + [i]
                score = cross_val_score(model, X[:, temp], y, cv=5).mean()
                scores.append(score)
            else:
                scores.append(-np.inf)

        best = np.argmax(scores)

        # early stopping based on improvement
        if scores[best] > best_score or force_full:
            selected.append(best)
            best_score = scores[best]
        else:
            break

    print("\nSelected feature indices:", selected)
    print("Selected feature names:", [feature_names[i] for i in selected])
    return selected

# ✅ Load dataset
df = pd.read_csv("data/Faults.csv")
X_all = df.drop(columns=['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'])
y_all = df['Stains']
feature_names = X_all.columns.tolist()

# ✅ Standardize
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# ✅ Feature selection (force_full=False → early stop, True → exactly 8)
selected_indices = forward_selection_fast(X_all_scaled, y_all, max_features=8, feature_names=feature_names, force_full=False)

# ✅ Use selected features for final training/testing
X_selected = X_all_scaled[:, selected_indices]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_all, test_size=0.2, random_state=42)

# ✅ Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# ✅ Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nNaive Bayes Accuracy on selected features:", round(accuracy * 100, 1), "%")
print("End Time:", time.process_time())
