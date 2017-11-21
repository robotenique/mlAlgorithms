import numpy as np
import numpy.linalg as LA
import math as m


def gaussianKernel(x1, x2, sigma):
    """returns a gaussian kernel between x1 and x2
    and returns the value in sim
    """
    # RBF Kernel
    return m.exp(-(LA.norm(x1.ravel() - x2.ravel())**2)/(2*sigma**2))
