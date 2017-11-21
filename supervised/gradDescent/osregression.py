import numpy as np
from sklearn import linear_model
from matplotlib import use, cm
use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math

def main():
    data = np.loadtxt('osdata.txt', delimiter=',')
    print(data)
    X = data[:, :3]
    X = np.column_stack((X, X[:, 2]**2))
    y = data[:, 3]
    m = y.T.size
    regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
    regr.fit(X, y)

    print('Theta found: ')
    print('%s %s %s %s\n' % (regr.coef_[0], regr.coef_[1],regr.coef_[2], regr.coef_[3]))
    #theta learned = 0.195797846404 1.14970377455 0.0370065601353
    print("Predição P(t0, dt, Punctuality, Punctuality²):")
    predict([0, 3, 0.5, 0.5**2],regr)
    predict([10, 30, 4, 4**2],regr)
    predict([20, 20, 20, 20**2],regr)
    predict([20, 3, 10, 10**2],regr)
    # Surface plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    plotPriority(regr, 400, -100, fig, ax, cm.magma)
    plotPriority(regr, 400, 50, fig, ax, cm.plasma)
    plotPriority(regr, 400, 200, fig, ax, cm.viridis)
    plotQuantumF()
    plt.show()



def plotPriority(regr, iterations, t0, fig, ax, cmm):
    #=========== Visualizing P(dt, Punctuality) =================#
    print("Calculating P(dt,Punctuality) plot...")
    # Grid over which we will calculate J
    dt_vals = np.linspace(1, 100, iterations)
    punc_vals = np.linspace(0, 100, iterations)
    # Priority matrix, initialized with zeros
    P_vals=np.array(np.zeros(iterations).T)
    for i in range(dt_vals.size):
        col = []
        for j in range(punc_vals.size):
            t = np.array([dt_vals[i],punc_vals[j]])
            val = np.array([t0, dt_vals[i], punc_vals[j], punc_vals[j]**2]).dot(regr.coef_)
            val = 1 if val < 1 else val
            col.append(val)
        P_vals=np.column_stack((P_vals,col))
    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    P_vals = P_vals[:,1:].T
    dt_vals, punc_vals = np.meshgrid(dt_vals, punc_vals)

    ax.plot_surface(dt_vals, punc_vals, P_vals, rstride=8, cstride=8, alpha=0.3,
                    cmap=cmm, linewidth=0, antialiased=False)
    ax.set_xlabel('dt')
    ax.set_ylabel('Punctuality')
    ax.set_zlabel('P(dt, Punctuality)')

def predict(arr, regr):
        predict = np.array(arr).dot(regr.coef_)
        print(f'For entry {arr[0]},{arr[1]},{arr[2]},{arr[3]} = {predict}')

def plotQuantumF():
    a = -0.676305221
    b = 20.676305221
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    x = np.linspace(0,280,50)
    y = list(map(lambda x :-67*math.log( 1/(1+math.exp(-x/47)),10), x))
    ax.plot(x,y,"r-")
    ax.set_xlabel('Priority')
    ax.set_xlim(0,300)
    ax.set_ylim(0,21)
    ax.set_ylabel('Quantum multiplier')



if __name__ == '__main__':
    main()
