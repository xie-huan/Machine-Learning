import numpy as np

def accuracy_score(y_true, y_predict):
    return sum(y_predict == y_true) / len(y_true)