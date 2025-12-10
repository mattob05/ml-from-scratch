import numpy as np
from copy import deepcopy
from model_selection import KFold

def cross_val_score(estimator, X, y, scoring=None, cv=5):
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)

    scores = []
    for train_ind, test_ind in cv.split(X):
        estimator_clone = deepcopy(estimator)
        X_train = X[train_ind]
        X_test = X[test_ind]
        y_train = y[train_ind]
        y_test = y[test_ind]
        estimator_clone.fit(X_train, y_train)
        if scoring is None:
            scores.append(estimator_clone.score(X_test, y_test))
        else:
            y_pred = estimator_clone.predict(X_test)
            scores.append(scoring(y_test, y_pred))

    return scores




