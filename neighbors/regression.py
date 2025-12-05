import numpy as np

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5, weights='uniform', p=2, metric='minkowski'):
        self.neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.metric = metric
        self.points = None
        self.labels = None

    def fit(self, X, y):
        self.points = X
        self.labels = y

    def predict(self, X):
        X_brod = X[:, np.newaxis, :]
        points_brod = self.points[np.newaxis, :, :]
        if self.metric == 'minkowski':
            distances = np.sum(np.abs(X_brod - points_brod)**self.p, axis=2) ** (1 / self.p)
        elif self.metric == 'euclidean':
            distances = np.sqrt(np.sum((X_brod - points_brod)**2, axis=2))
        else: # manhattan distance
            distances = np.sum(np.abs(X_brod - points_brod), axis=2)

        closest = np.argpartition(distances, self.neighbors, axis=1)[:, :self.neighbors]
        dist_clos = distances[np.arange(distances.shape[0])[:, None], closest]
        lab_clos = self.labels[closest]
        
        if self.weights == 'distance':
            w = 1 / (dist_clos + 1e-12) #adding epsilon to avoid division by zero
        else:
            w = np.ones_like(dist_clos)

        weighted_sum = np.sum(lab_clos * w, axis=1)
        total_weight = np.sum(w, axis=1)

        return np.divide(weighted_sum, total_weight, out=np.zeros_like(weighted_sum), where=total_weight!=0)

        


