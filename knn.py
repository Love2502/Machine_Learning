import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import time

# Start timing
print("Start Time:", time.process_time())

#1. Define custom KNN predictor
def knn_predict(X_train, y_train, x_test, k=3):
    distances = []
    for i in range(len(X_train)):
        distance = np.sqrt(np.sum((x_test - X_train[i]) ** 2))
        label = y_train.iloc[i]
        distances.append((distance, label))
    distances.sort(key=lambda x: x[0])
    top_k = [label for _, label in distances[:k]]
    return max(set(top_k), key=top_k.count)

#2. Fast forward selection using sklearn KNN
def forward_selection_fast(X, y, max_features, feature_names):
    selected = []
    best_score = -np.inf
    knn = KNeighborsClassifier(n_neighbors=5)

    while len(selected) < max_features:
        scores = []
        for i in range(X.shape[1]):
            if i not in selected:
                temp = selected + [i]
                score = cross_val_score(knn, X[:, temp], y, cv=5).mean()
                scores.append(score)
            else:
                scores.append(-np.inf)
        best = np.argmax(scores)
        if scores[best] > best_score:
            selected.append(best)
            best_score = scores[best]
        else:
            break

    print("\nSelected feature indices:", selected)
    print("Selected feature names:", [feature_names[i] for i in selected])
    return selected

#3. Load and prepare data
df = pd.read_csv("data/Faults.csv")
X_all = df.drop(columns=['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'])
y_all = df['Pastry']
feature_names = X_all.columns.tolist()

#4. Standardize features
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

#5. Forward selection
selected_indices = forward_selection_fast(X_all_scaled, y_all, max_features=8, feature_names=feature_names)

#6. Split with only selected features
X_selected = X_all_scaled[:, selected_indices]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_all, test_size=0.2, random_state=42)

#7. Final prediction with custom KNN
# print("\nRunning KNN on selected features...")
predictions = [knn_predict(X_train, y_train, x, k=5) for x in X_test]
accuracy = sum([predictions[i] == y_test.iloc[i] for i in range(len(y_test))]) / len(y_test)

#8. Output results
print("\nKNN Accuracy on selected features:", round(accuracy * 100, 2), "%")
print("End Time:", time.process_time())
