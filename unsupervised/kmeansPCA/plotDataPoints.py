import matplotlib.pyplot as plt
from show import show


def plotDataPoints(X, idx):
    """plots data points in X, coloring them so that those
    with the same index assignments in idx have the same color
    """
    # Create palette
    # palette = hsv(K + 1)
    # colors = palette(idx, :)
    #
    # # Plot the data

    cmap = plt.get_cmap("jet")
    idxn = idx.astype('float') / max(idx.astype('float'))
    colors = cmap(idxn)
    plt.scatter(X[:, 0], X[:, 1], 15, edgecolors=colors,
                marker='o', facecolors='none', lw=0.5)
