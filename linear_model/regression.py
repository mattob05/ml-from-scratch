import numpy as np

class LinearRegression:
    def __init__(self, method='ols', learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.lr = learning_rate
        self.iter = n_iterations
        self.coef = None
        self.intercept = None
    
    def fit(self, X, y):
        if self.method == 'ols':
            X_biased = np.c_[np.ones(len(X)), X]
            # Using the equation (X^T * X) * w = X^T * y
            a = np.dot(X_biased.T, X_biased)
            b = np.dot(X_biased.T, y)
            # Solve for a * w = b
            w = np.linalg.solve(a, b)
            self.intercept = w[0]
            self.coef = w[1:]
        else:
            n_samples, n_features = X.shape
            self.coef = np.zeros(n_features)
            self.intercept = 0

            for _ in range(self.iter):
                y_pred = np.dot(X, self.coef) + self.intercept
                # Calculate partial derivates
                dc = (1 / n_samples) * np.dot(X.T, y_pred - y)
                di = (1 / n_samples) * np.sum(y_pred - y)
                # Update coefficients
                self.coef -= self.lr * dc
                self.intercept -= self.lr * di

        return self
    
    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept
