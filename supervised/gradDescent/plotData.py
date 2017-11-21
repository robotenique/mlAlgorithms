import matplotlib.pyplot as plt
import numpy as np

def plotData(data, show=True):
    """
    plots the data points and gives the figure axes labels of
    population and profit.
    """
    plt.title("Population x Profit")
    plt.xlabel("City population (x 10000)")
    plt.ylabel("Profit (x $10000)")
    plt.plot(data[:, 0], data[:, 1], 'rx', markersize=10)
    if show:
        plt.show()
