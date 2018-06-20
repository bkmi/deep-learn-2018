import random

import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.linalg import fractional_matrix_power


def load(verbose=False):
    with np.load('dimredux-challenge-01-data.npz') as fh:
        data_x = fh['data_x']
        validation_x = fh['validation_x']
        validation_y = fh['validation_y']

    if verbose:
        print('data_x: ', data_x.shape, data_x.dtype)
        print('validation_x: ', validation_x.shape, validation_x.dtype)
        print('validation_y: ', validation_y.shape, validation_y.dtype)

    return data_x, validation_x, validation_y


def lag_data(data, lag=0):
    """Lags data on axis=0."""
    assert data.shape[0] > lag, 'you need more samples than lag'
    assert lag >= 0, 'you need a non-negative lagtime'

    if lag == 0:
        not_lagged = data
        lagged = data
    else:
        not_lagged = data[:-lag, ...]
        lagged = data[lag:, ...]
    return not_lagged, lagged


def pca(x, remove_mean=True, whiten=True, axis=0):
    if remove_mean:
        x = x - x.mean(axis=axis)
    return PCA(whiten=whiten).fit_transform(x)


def whiten(x, axis=0):
    x = x - x.mean(axis=axis)
    n = np.take(x.shape, axis)
    cxx = fractional_matrix_power(np.matmul(x.T, x) / n, -1/2)
    return np.tensordot(x, cxx, axes=(1, 0))


def cluster_compare(true_states, predicted_states):
    kmeans = KMeans(n_clusters=4, random_state=0)
    clustered = kmeans.fit(predicted_states)
    return adjusted_rand_score(true_states, clustered)


def shuffle(x):
    y = x.copy()
    random.shuffle(y)
    return y
