import numpy as np


def polyFeatures(X, p):
    """takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    """
    # You need to return the following variables correctly.
    X_poly = X.reshape(X.size, 1)
    m = X_poly[:, 0].size
    for i in range(p):
        next_pow = (X_poly[:, X_poly.shape[1] - 1]*X_poly[:, 0]).reshape(m, 1)
        X_poly = np.concatenate((X_poly, next_pow), axis=1)

    return X_poly
