# Steel Plates Fault Detection

## Project Overview

This project applies machine learning techniques to detect faults in steel plates using a dataset that contains 27 numeric input features and 7 binary fault-type indicators.

Currently, the implementation includes a custom k-Nearest Neighbors (KNN) classifier. Future versions will incorporate additional models such as Decision Trees, Logistic Regression, and Support Vector Machines (SVM(maybe)) to enable comparative analysis and performance evaluation.

## Dataset Description

- Source: Steel Plates Faults Dataset (UCI Machine Learning Repository)
- Records: 1,941 samples
- Features:
  - 27 numerical input variables
  - 7 binary columns representing distinct fault types
- Objective: Perform binary classification to predict the presence of a specific fault type (e.g., 'Pastry')

## Project Structure

```
├── data/
│   ├── Faults.NNA              # Raw dataset with no column names
│   └── Faults27x7_var          # List of column names
│   └── Faults.csv              # The complete dataset with the column names
├── main.py                     # Script containing model implementation and evaluation
├── README.md                   # Project documentation
```

## Requirements

Python version 3.7 or above is required.

### Required Python Packages

- numpy
- pandas
- matplotlib
- seaborn
- ucimlrepo (needed to download the dataset if it doesn't exist)

You can install these packages using pip:


### Creating a Virtual Environment (Recommended)

```bash
  python -m venv .venv

  # Then Activate the environment:

  # On Windows:
  venv\Scripts\activate
  or 
  .venv/Scripts/Activate.ps1

  # On macOS/Linux:
  source .venv/bin/activate

  # Install dependences
  pip install -r requirements.txt
```


## How to Run

1. Ensure the dataset files (`Faults.NNA` and `Faults27x7_var`) are placed inside a `data/` directory. Otherwise install the `ucimlrepo` library.

2. Run the main script using:

```bash
  python main.py
```

This will:
- Load and preprocess the dataset
- Train and test a machine learning model (initially KNN)
- Output model accuracy
- Generate a visualization of predictions and a confusion matrix

## Planned Enhancements

- Support for multiple classifiers (Decision Tree, Logistic Regression, SVM(maybe))
- Hyperparameter tuning and model selection
- Exportable performance reports

## Academic Use

This project has been developed for educational purposes in a university setting. All machine learning algorithms are implemented from scratch to provide clarity and a deeper understanding of their internal workings.


## Team Information

**Team Members:**
- Jamal Dassrath (Immatriculation No.: 22301035)
- Love - (Immatriculation No.: 12306406)
- Muhammad Ahtisham Bhatti (Immatriculation No.: 22301502)

**Study Course:** Artificial Intelligence B.Sc.  
**Semester:** 4th Semester
