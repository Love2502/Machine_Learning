# preprocessing.py

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def load_preprocessed_data(fault_type='Stains', k_features=6):
    data = pd.read_csv("data/Faults.csv")

    X = data.iloc[:, :-7]
    y = data[fault_type]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_best = selector.fit_transform(X_scaled, y)

    total = len(X_best)
    split = int(0.8 * total)

    X_train = X_best[:split]
    X_test = X_best[split:]
    y_train = y[:split]
    y_test = y[split:]

    return X_train, X_test, y_train, y_test



