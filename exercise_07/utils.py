import numpy as np


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
    data = data[:lag]
    lagged = data[lag:]
    return data, lagged


data_x, validation_x, validataion_y = load(verbose=True)
