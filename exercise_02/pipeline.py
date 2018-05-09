import numpy as np


def loader():
    with np.load('prediction-challenge-01-data.npz') as f:
        X = f['data_x']
        y = f['data_y']
        X_test = f['test_x']
    return X, y, X_test


def saver(prediction):
    # MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
    assert prediction.ndim == 1
    assert prediction.shape[0] == 2000

    # AND SAVE EXACTLY AS SHOWN BELOW
    np.save('prediction.npy', prediction)

if __name__ == '__main__':
    pass