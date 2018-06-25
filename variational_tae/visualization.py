import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def plot_labeled_3d(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for state, (c, m) in enumerate([('r', 'o'), ('b', '^'), ('y', 's'), ('g', '*')]):
        state_mask = labels == state
        data_subset = data[state_mask]
        ax.scatter(data_subset[:, 0],
                   data_subset[:, 1],
                   data_subset[:, 2],
                   c=c,
                   marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

import utils

x, vx, vy = utils.load()
plot_labeled_3d(vx, vy)
