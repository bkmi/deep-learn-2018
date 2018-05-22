import numpy as np


def loader():
    with np.load('prediction-challenge-02-data.npz') as f:
        X = f['data_x']
        y = f['data_y']
        X_test = f['test_x']
    # TRAINING DATA: INPUT (x) AND OUTPUT (y)
    # 1. INDEX: IMAGE SERIAL NUMBER (6000)
    # 2. INDEX: COLOR CHANNELS (3)
    # 3/4. INDEX: PIXEL VALUE (32 x 32)
    print(data_x.shape, data_x.dtype)
    print(data_y.shape, data_y.dtype)
    
    # TEST DATA: INPUT (x) ONLY
    print(test_x.shape, test_x.dtype)
    return X, y, X_test


def saver(prediction):
    # MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
    assert prediction.ndim == 1
    assert prediction.shape[0] == 300

    # AND SAVE EXACTLY AS SHOWN BELOW
    np.save('prediction.npy', prediction)

if __name__ == '__main__':
    pass
