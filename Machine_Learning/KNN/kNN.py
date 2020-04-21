import numpy as np
from collections import Counter
from math import sqrt
from .metrics import accuracy_score

# def kNN_classify(k,X_train,y_train,x):
#     # distances = []
#     #     for x_train in X_train:
#     #         dis = sqrt(np.sum((x_train - x)**2))
#     #         distances.append(dis)
#
#     distances = np.array([sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train])
#
#     topK_y = y_train[np.argsort(distances)[:k]]
#
#     votes = Counter(topK_y)
#
#     predict_y = votes.most_common(1)[0][0]
#     return predict_y

class KNNClassifier:

    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]

        return np.array(y_predict)

    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        topK_y = self._y_train[np.argsort(distances)[:self.k]]
        votes = Counter(topK_y)

        predict_y = votes.most_common(1)[0][0]
        return predict_y

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test,y_predict)

    def __repr__(self):
        return "kNN(k = %d)" % self.k
