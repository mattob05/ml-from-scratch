import numpy as np

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tp, fp], [fn, tn]])

def accuracy_score(y_true, y_pred):
    conf_mx = confusion_matrix(y_true, y_pred)
    return (conf_mx[0, 0] + conf_mx[1, 1]) / len(y_pred)

def precision_score(y_true, y_pred):
    conf_mx = confusion_matrix(y_true, y_pred)
    denom = (conf_mx[0, 0] + conf_mx[0, 1])
    if denom == 0: return 0
    return conf_mx[0, 0] / denom

def recall_score(y_true, y_pred):
    conf_mx = confusion_matrix(y_true, y_pred)
    denom = (conf_mx[0, 0] + conf_mx[1, 0])
    if denom == 0: return 0
    return conf_mx[0, 0] / denom

def specificity_score(y_true, y_pred):
    conf_mx = confusion_matrix(y_true, y_pred)
    denom = (conf_mx[1, 1] + conf_mx[0, 1])
    if denom == 0: return 0
    return conf_mx[1, 1] / denom

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
