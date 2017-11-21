# Logistic Regression
from matplotlib import use

use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pandas as pd

from ml import mapFeature, plotData, plotDecisionBoundary
from matplotlib.pyplot import show
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from sigmoid import sigmoid
from predict import predict


def optimize(Lambda):

    result = minimize(costFunctionReg, initial_theta, method='L-BFGS-B',
                      jac=gradientFunctionReg, args=(X, y, Lambda),
                      options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

    return result


# Plot Boundary
def plotBoundary(theta, X, y):
    plotDecisionBoundary(theta, X, y)
    plt.title(r'$\lambda$ = ' + str(Lambda))

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    show()


# Initialization

# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).
plt.figure(figsize=(15, 10))
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

plotData(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
show()
input('Program paused. Press Enter to continue...')


# =========== Part 1: Regularized Logistic Regression ============

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = pd.DataFrame(X)
X = X.apply(mapFeature, axis=1)
# convert back to numpy ndarray
X = X.values

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
Lambda = 0.0

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, Lambda)

print('Cost at initial theta (zeros): %f' % cost)

# ============= Part 2: Regularization and Accuracies =============

# Optimize and plot boundary

Lambda = 1.0
result = optimize(Lambda)
theta = result.x
cost = result.fun

# Print to screen
print('lambda = ' + str(Lambda))
print('Cost at theta found by scipy: %f' % cost)
print('theta:', ["%0.4f" % i for i in theta])

input('Program paused. Press Enter to continue...')

plotBoundary(theta, X, y)

# Compute accuracy on our training set
p = predict(theta, X)
acc = np.mean(np.where(p == y, 1, 0)) * 100
print('Train Accuracy: %f' % acc)

input('Program paused. Press Enter to continue...')


# ============= Part 3: Different values of lambda =============

for Lambda in np.linspace(0.0, 100.1, 8):
    result = optimize(Lambda)
    theta = result.x
    print('lambda = ' + str(Lambda))
    print('theta:', ["%0.4f" % i for i in theta])
    plotBoundary(theta, X, y)
input('Program paused. Press Enter to continue...')
