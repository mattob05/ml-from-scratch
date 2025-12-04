import numpy as np

class StandardScaler():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std[self.std == 0] = 1

        return self

    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.data_range = None
        self.range = feature_range


    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        
        self.data_range = self.max - self.min
        # protection against division by zero
        self.data_range[self.data_range == 0] = 1

        return self

    def transform(self, X):
        new_X = (X - self.min) / self.data_range
        if self.range != (0, 1):
            a, b = self.range
            new_X = new_X * (b-a) + a
        return new_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class RobustScaler():
    def __init__(self):
        self.median = None
        self.q1 = None
        self.q3 = None
        self.iqr = None

    def fit(self, X):
        self.median = np.median(X, axis=0)
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        
        self.iqr = self.q3 - self.q1
        self.iqr[self.iqr == 0] = 1

        return self
    
    def transform(self, X):
        return (X - self.median) / self.iqr

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
