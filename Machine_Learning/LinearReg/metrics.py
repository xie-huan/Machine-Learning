import numpy as np


def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), "length of y_true and y_predict should equal"
    mse = (y_true - y_predict).dot(y_true - y_predict) / len(y_true)
    return mse


def root_mean_squared_error(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), "length of y_true and y_predict should equal"
    mae = np.sum(np.abs(y_true - y_predict)) / len(y_true)
    return mae


def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true,y_predict) / np.var(y_true)
