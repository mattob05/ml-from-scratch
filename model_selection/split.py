import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    X_arr = np.array(X)
    y_arr = np.array(y)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    train_indices = []
    test_indices = []
    
    if stratify is not None:
        strat_arr = np.array(stratify)
        classes = np.unique(stratify)

        for cls in classes:
            cls_indices = np.where(strat_arr == cls)[0]

            if shuffle:
                np.random.shuffle(cls_indices)
            
            n_test_cls = int(len(cls_indices) * test_size)
            test_indices.extend(cls_indices[:n_test_cls])
            train_indices.extend(cls_indices[n_test_cls:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

    else:
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        n_test = int(n_samples * test_size)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    return X_arr[train_indices], X_arr[test_indices], y_arr[train_indices], y_arr[test_indices]


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = n_samples // self.splits
        splitted = np.split(indices, fold_sizes)
        for i in range(fold_sizes):
            test_indices = splitted[i]
            train_indices = np.concatenate(np.delete(splitted, i))
            yield train_indices, test_indices
        




        