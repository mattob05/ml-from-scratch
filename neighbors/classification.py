import numpy as np
from metrics import accuracy_score

class KNeighborsClassifier:
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

        return self

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
        pred = []
        
        for i in range(X.shape[0]):
            d = dist_clos[i]
            lab = lab_clos[i]
            # to avoid dividing by zero
            if_zero = np.where(d == 0)[0]
            if len(if_zero) > 0:
                pred.append(lab[if_zero[0]])
                continue

            if self.weights == 'distance':
                w = 1 / d
            else:
                w = np.ones_like(d)

            classes = np.unique(lab)
            scores = {c: 0 for c in classes}

            for weight, lbl in zip(w, lab):
                scores[lbl] += weight

            best_class = max(scores, key=scores.get)
            pred.append(best_class)

        return np.array(pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


        


