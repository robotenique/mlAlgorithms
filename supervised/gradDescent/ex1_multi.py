import numpy as np
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
from featureNormalize import featureNormalize
from featureNormalize import normEntry
# ================ Part 1: Feature Normalization ================

print('Loading data ...')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.T.size


# Print out some data points
print('First 10 examples from the dataset:')
print(np.column_stack( (X[:10], y[:10]) ))
input("Program paused. Press Enter to continue...")

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)
print('[mu] [sigma]')
print(mu, sigma)


# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), axis=1)

print('Running gradient descent ...')
# Choose some alpha value
alpha = 0.01
num_iters = 9000
# Init Theta and Run Gradient Descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
# Plot the convergence graph
plt.plot(J_history, '-b')

''' #Plot for some different alphas
plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
for alpha in np.linspace(0.001, 0.6, 10):
    num_iters = 500
    # Init Theta and Run Gradient Descent
    theta = np.zeros(3)
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
    # Plot the convergence graph
    plt.plot(J_history, '-b')
'''
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
input("Program paused. Press Enter to continue...")

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
nPrice = normEntry([1, 1650, 3], mu, sigma)
price = np.array(nPrice).dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house')
print('(using gradient descent): ')
print(price)

input("Program paused. Press Enter to continue...")

# ================ Part 3: Normal Equations ================

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

print('Solving with normal equations...')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.T.size

# Add intercept term to X
X = np.concatenate((np.ones((m,1)), X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(' %s \n' % theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, 1650, 3]).dot(theta)

# ============================================================

print("Predicted price of a 1650 sq-ft, 3 br house ")
print('(using normal equations):\n $%f\n' % price)

input("Program paused. Press Enter to continue...")
