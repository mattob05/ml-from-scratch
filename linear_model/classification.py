import numpy as np
from metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.iter = n_iterations
        self.coef = None
        self.intercept = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.iter):
            linear = np.dot(X, self.coef) + self.intercept
            y_pred = self.sigmoid(linear)

            dc = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            di = (1 / n_samples) * np.sum(y_pred - y)

            self.coef -= self.lr * dc
            self.intercept -= self.lr * di

        return self
    
    def predict(self, X, threshold=0.5):
        linear = np.dot(X, self.coef) + self.intercept
        y_pred = self.sigmoid(linear)

        return (y_pred > threshold).astype(int)

    def predict_proba(self, X):
        linear = np.dot(X, self.coef) + self.intercept
        return self.sigmoid(linear)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
