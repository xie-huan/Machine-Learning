import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):

    # 设置seed参数为了方便调试
    # 因为在代码中含有随机数生成
    # 想让两次调用函数产生的随机数一样，就设置seed为固定值
    # 否则，不设置seed的值

    if seed:
        np.random.seed(seed)

    shuffle_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    train_indexes = shuffle_indexes[:len(X) - test_size]
    test_indexes = shuffle_indexes[len(X) - test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
