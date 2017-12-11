import matplotlib.pyplot as plt

from plotDataPoints import plotDataPoints
from show import show


def plotProgresskMeans(X, centroids, previous, idx, K, i, color, ax):
    """plots the data
    points with colors assigned to each centroid. With the previous
    centroids, it also plots a line between the previous locations and
    current locations of the centroids.
    """

    # Plot the examples
    plotDataPoints(X, idx)

    # Plot the centroids as black x's
    ax.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=60, lw=3, color='black')

    # Plot the history of the centroids with lines
    for j in range(len(centroids)):
        ax.plot([centroids[j, 0], previous[j, 0]],
                [centroids[j, 1], previous[j, 1]], color='black')

    # Title
    plt.title('Iteration number %d' % i)
    input('Program paused. Press Enter to continue...')

