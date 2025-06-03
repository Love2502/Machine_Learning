import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file_path = os.path.join("data", "Faults.csv")

# ============================================================================================
# ==================================== Data Loading ==========================================
# ============================================================================================

# Checks if dataset file exists
if os.path.exists(file_path):
    print("Loading dataset into memory")
    steel_data = pd.read_csv(file_path)
    
else: # If not we download it
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


# # Randomize the dataset
# steel_data = steel_data.sample(frac=1, random_state=12).reset_index(drop=True) # Not neccesarry as we can randomize when doing the test-train split

# ============================================================================================
#                            Exploratory Data Analysis
# ============================================================================================

print("\n================= Exploring the data =================")
print(steel_data.describe())
print(steel_data.head())
print(steel_data.info())


steel_classes = steel_data[steel_data.columns[-7:]]

# Checks if the data is ordered by classes
label_series = steel_classes.idxmax(axis=1)
plt.figure(figsize=(14, 4))
plt.plot(label_series.reset_index(drop=True), marker='.', linestyle='none')
plt.title('Class Distribution Over Row Index')
plt.ylabel('Class')
plt.xlabel('Row Index')
plt.xticks(ticks=range(0, len(label_series), 100))
plt.show()

class_counts = steel_classes.sum()

# Bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index, class_counts.values)

plt.title('Steel Fault Class Distribution')
plt.xlabel('Fault Class')
plt.ylabel('Number of Instances')

# Add numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# Separate features and targets
# Assuming the last few columns are fault types (binary targets)
X = steel_data.iloc[:, :-7]  # adjust depending on actual target columns
y = steel_data.iloc[:, -7:]  # 7 target fault types
    
# =============================================================================
#                            Feature Selection
# =============================================================================
print("\n================= Selecting Features =================")
    
# Tests different sets of features and checks the accuracy on the KNN Classifier 
def trainAndTest(selected_features): 
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.3, random_state=23)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Step 3: Predict
    y_pred = knn.predict(X_test_scaled)

    # Step 4: Convert one-hot to class labels
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Step 5: Confusion Matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, normalize='true')
    avg_diag = np.trace(conf_matrix) / conf_matrix.shape[0]
    
    # Format the output with fixed width for alignment (Not important)
    features_str = str(selected_features).ljust(150)  
    print(f"Features: {features_str} Accuracy: {avg_diag:>7.4f}")  
    
    return avg_diag


def forward_feature_selection(all_features, MAXF):
    selected = []
    selected_last_iter = []
    c_rate_best = None
    
    while len(selected) < MAXF:
        selected_last_iter = selected.copy()
        c_rate = [0] * len(all_features)
        
        for i, feature in enumerate(all_features):
            if feature not in selected:
                selected_temp = selected + [feature]
                c_rate[i] = trainAndTest(selected_temp)
        
        x_best_addition = c_rate.index(max(c_rate))
        
        if c_rate[x_best_addition] > (c_rate_best if c_rate_best is not None else -1):
            selected.append(all_features[x_best_addition])
            c_rate_best = c_rate[x_best_addition]
        else:
            break
    
    return selected


# Example list of features (replace with your actual features)
all_features = list(X)
MAXF = 5  # Maximum number of features to select

selected_features = forward_feature_selection(all_features, MAXF)
print("Selected features: ", selected_features)

X = steel_data[list(selected_features)]

# Make a new df with the fewer features
df_features_selected = pd.concat([X,y], axis=1)


# =============================================================================
#                            Data Cleaning
# =============================================================================

# print("\n================= Removing the Outlier =================")

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
# y = df_cleaned.iloc[:, -7:]


# =============================================================================
#                            Data Splitting
# =============================================================================
print("\n================= Splitting the Data =================")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)


# Check shapes
print("Train features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Train labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)

# =============================================================================
#                            Data Scaling / Normalization
# =============================================================================
print("\n================= Scaling / Normalizing Data =================")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data was standardized")


# ************************************************************************************
# *************************** Model Evaluation + Parameter Tuning ********************
# ************************************************************************************

for k in range(1,10):
    # Step 2: Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Step 3: Predict
    y_pred = knn.predict(X_test_scaled)

    # Step 4: Convert one-hot to class labels
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Step 5: Confusion Matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, normalize='true')
    avg_diag = np.trace(conf_matrix) / conf_matrix.shape[0]
    
    # Step 6: Output
    print(f"Average of diagonal (mean class-wise accuracy) for KNN with k = {k}: {avg_diag:.4f}")
    
    bestK = 0
    bestAcc = 0.0
    if avg_diag > bestAcc:
        bestK, bestAcc = k, avg_diag



print(f"After iterating through the values of K, the best ones is {bestK}")


knn = KNeighborsClassifier(n_neighbors=bestK)
knn.fit(X_train_scaled, y_train)


y_pred = knn.predict(X_test_scaled)


y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)


conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, normalize='true')
avg_diag = np.trace(conf_matrix) / conf_matrix.shape[0]


print(f"Average of diagonal (mean class-wise accuracy) for KNN with k = {bestK}: {avg_diag:.4f}")


disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title(f"Relative Confusion Matrix for KNN Classifier for {bestK}")
plt.show()