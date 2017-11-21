import sys
sys.path.append("../")
from nnMultiClass.mlCalcs import sigmoid



def sigmoidGradient(z):

    return sigmoid(z)*(1 - sigmoid(z))
