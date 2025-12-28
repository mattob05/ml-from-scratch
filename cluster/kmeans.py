import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, init='random', n_init='auto', max_iter=300, tol=0.0001, random_state=None):
        self.clusters = n_clusters
        self.init = init
        if n_init == 'auto':
            if init == 'random':
                self.n_init = 10
            else:
                self.n_init = 1
        else:
            self.n_init = n_init    
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.clusters_dist = None


    def _init_centroids(self, X):
        np.random.seed(self.random_state)
        if self.init == 'random':
            indices = self.rng.choice(len(X), size=self.clusters, replace=False)
            self.centroids = X[indices]
        else:
            # kmeans++
            self.centroids = []
            index = self.rng.randint(0, len(X))
            self.centroids.append(X[index])
            min_sq_distances = np.array([np.inf]*len(X))

            for _ in range(self.clusters-1):
                squared = np.pow(np.subtract(X, self.centroids[-1]), 2)
                sum = np.sum(squared, axis=1)
                min_sq_distances = np.minimum(sum, min_sq_distances)
                proba = min_sq_distances / (np.sum(min_sq_distances) + 1e-10)
                next_centroid_idx = np.random.choice(len(X), p=proba)
                self.centroids.append(X[next_centroid_idx])

        self.centroids = np.array(self.centroids)

    
    def _assign_labels(self, X, if_fit=True):
        # calculating euclidian distance without square root because of numerical cost
        squared = np.pow(np.subtract(X[:, np.newaxis, :], self.centroids[np.newaxis, :, :]), 2)
        dist = np.sum(squared, axis=2)
        labels = np.argmin(dist, axis=1)
        clusters_dist = np.min(dist, axis=1)
        inertia = np.sum(clusters_dist)
        if if_fit:
            self.labels = labels
            self.clusters_dist = clusters_dist
            self.inertia = inertia
        else:
            return labels, inertia


    def _update_centroids(self, X):
        for i in range(self.clusters):
            cluster_points = X[self.labels == i]
            if cluster_points.shape[0] == 0:
                # we search for the point that is furthest from it's cluster and set it as new center
                idx = np.argmax(self.clusters_dist)
                self.centroids[i] = X[idx]
                self.clusters_dist[idx] = 0
            else:
                new_center = np.mean(cluster_points, axis=0)
                self.centroids[i] = new_center


    def fit(self, X, y=None):
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        for _ in range(self.n_init):
            self._init_centroids(X)
            self.labels = None
            for _ in range(self.max_iter):
                old_labels = self.labels
                old_centroids = self.centroids.copy()
                
                self._assign_labels(X)
                if old_labels is not None and np.array_equal(old_labels, self.labels):
                    break
                self._update_centroids(X)
                
                shift = np.sum(np.pow(self.centroids - old_centroids, 2))
                if shift < self.tol:
                    break
            if self.inertia < best_inertia:
                best_inertia = self.inertia
                best_centers = self.centroids.copy()
                best_labels = self.labels.copy()
        
        self.inertia = best_inertia
        self.labels = best_labels
        self.centroids = best_centers

        return self
    

    def predict(self, X):
        labels, _ = self._assign_labels(X, if_fit=False)
        return labels
    
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


    def score(self, X):
        _, inertia = self._assign_labels(X, if_fit=False)
        return -inertia
    

    def transform(self, X):
        squared = np.pow(np.subtract(X[:, np.newaxis, :], self.centroids[np.newaxis, :, :]), 2)
        dist = np.sum(squared, axis=2)
        return np.sqrt(dist)
    

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)