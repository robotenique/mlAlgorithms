import numpy as np
import sklearn.svm as svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """
    vals = np.array(list(map(lambda a: 0.01*3**a, range(7))))
    best_C = 0
    best_sigma = 0
    best_val = -1
    gamma = lambda sigma : 1/(2*sigma**2)
    for C in vals:
        for sigma in vals:
            clf = svm.SVC(C=C, kernel='rbf', tol=1e-3, max_iter=2000, gamma=gamma(sigma))
            model = clf.fit(X, y)
            acc = np.sum(np.where(model.predict(Xval) == yval, 1, 0))/len(yval)
            print(acc)
            if acc > best_val:
                best_val = acc
                best_C = C
                best_sigma = sigma

    return best_C, best_sigma
