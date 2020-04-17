import numpy as np

class StandardScalar:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self,X):
        # 只处理二维
        # [
        #  [1,1,1,1,1],
        #  [2,2,2,2,2],
        #  [....],
        #  [....]
        # ]
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self,X):
        assert X.ndim == 2, "the dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None,"must fit before transform"
        assert X.shape[1] == len(self.mean_),"the future number of X must equal to the length of mean_ "

        resX = np.empty(shape=X.shape, dtype=float)

        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]
        return resX
