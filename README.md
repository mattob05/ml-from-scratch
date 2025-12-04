# Machine Learning from Scratch (NumPy Implementation) 🧠

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-numpy-orange)

An educational machine learning library implemented purely in **NumPy**.

The main goal of this project is to demystify the "black box" of popular ML algorithms (like Scikit-Learn) by building them from the ground up. It focuses on understanding the underlying mathematics, matrix operations, and optimization techniques.

## 🚀 Key Features

### 1. Linear Models
* **Linear Regression**:
    * **Gradient Descent Solver**: Iterative optimization with configurable learning rate and epochs.
    * **OLS Solver (Normal Equation)**: Analytical solution using linear system solving (more stable than explicit matrix inversion).

### 2. Preprocessing
Algorithms for feature scaling to improve model convergence:
* **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance.
* **MinMaxScaler**: Transforms features by scaling each feature to a given range (e.g., 0 to 1).
* **RobustScaler**: Scales features using statistics that are robust to outliers (Median and IQR).

### 3. Evaluation Metrics
Set of regression metrics:
* **MSE** (Mean Squared Error)
* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **RAE** (Relative Absolute Error)
* **RSE** (Relative Squared Error)
* **R2 Score** (Coefficient of Determination)
