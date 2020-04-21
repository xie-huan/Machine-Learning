import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        self.coef_ = None   # 系数
        self.interception_ = None    # 截距
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], ""

        #!!!important
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iter=1e4):

        assert X_train.shape[0] == y_train.shape[0], ""
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            # 向量化实现
            return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iter, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iter:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                i_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iter)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # n_iter 代表观测所有数据几次
    def fit_sgd(self, X_train, y_train, n_iter=5, t0=5, t1=50):

        assert X_train.shape[0] == y_train.shape[0], ""
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2

        # Stochastic gradient descent
        def sgd(X_b, y, initial_theta, n_iter, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            for curr_iter in range(n_iter):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(curr_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones([len(X_train), 1]), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iter)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def predict(self,X_predict):
        assert self.interception_ is not None and self.coef_ is not None,\
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_),\
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        y_predict = X_b.dot(self._theta)
        return y_predict

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression()"