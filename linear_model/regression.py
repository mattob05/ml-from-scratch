import numpy as np
from metrics import r2_score

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
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class RidgeRegression:
    def __init__(self, method='ols', alpha=1, learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.lr = learning_rate
        self.iter = n_iterations
        self.alpha = alpha
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef = np.zeros(n_features)
        self.intercept = 0
        if self.method == 'ols':
            X_biased = np.c_[np.ones(len(X)), X]
            I = np.eye(n_features + 1)
            I[0, 0] = 0 # as we don't regularize bias
            a = np.dot(X_biased.T, X_biased) + self.alpha * I
            b = np.dot(X_biased.T, y)
            w = np.linalg.solve(a, b)
            self.intercept = w[0]
            self.coef = w[1:]
        else:
            for _ in range(self.iter):
                y_pred = np.dot(X, self.coef) + self.intercept

                dc = (1 / n_samples) * np.dot(X.T, y_pred - y) + 2 * self.alpha * self.coef
                di = (1 / n_samples) * np.sum(y_pred - y)

                self.coef -= self.lr * dc
                self.intercept -= self.lr * di

        return self
    
    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    

class LassoRegression():
    def __init__(self, alpha=1, learning_rate=0.01, n_iterations=1000):
        self.alpha = alpha
        self.lr = learning_rate
        self.iter = n_iterations
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef = np.zeros(n_features)
        self.intercept = 0
        for _ in range(self.iter):
            y_pred = np.dot(X, self.coef) + self.intercept

            dc = (1 / n_samples) * np.dot(X.T, y_pred - y) + self.alpha * np.sign(self.coef)
            di = (1 / n_samples) * np.sum(y_pred - y)

            self.coef -= self.lr * dc
            self.intercept -= self.lr * di
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)