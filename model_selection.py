import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将矩阵X和标签y按照test_ration分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_train must be valid"

    if seed:
        np.random.seed(seed)

    shuffle_index = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_index = shuffle_index[:test_size]
    train_index = shuffle_index[test_size:]
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    return X_train, X_test, y_train, y_test

