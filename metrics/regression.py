import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mean_absolute_percentage_error(y_true, y_pred):
    # adding epsilon to avoid division by 0
    epsilon = 1e-10
    return np.mean(np.abs(y_true - y_pred) / y_true + epsilon) * 100

def relative_absolute_error(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    
    if den == 0:
        return np.nan
    
    return num/den

def relative_squared_error(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)

    if den == 0:
        return np.nan
    
    return num/den

def r2_score(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if den == 0:
        return np.nan
    
    return 1 - num/den
