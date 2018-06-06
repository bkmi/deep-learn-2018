import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def loader():
    with np.load('prediction-challenge-02-data.npz') as f:
        X = f['data_x']
        y = f['data_y']
        X_test = f['test_x']
    # TRAINING DATA: INPUT (x) AND OUTPUT (y)
    # 1. INDEX: IMAGE SERIAL NUMBER (6000)
    # 2. INDEX: COLOR CHANNELS (3)
    # 3/4. INDEX: PIXEL VALUE (32 x 32)
    print(X.shape, X.dtype)
    print(y.shape, y.dtype)
    
    # TEST DATA: INPUT (x) ONLY
    print(X_test.shape, X_test.dtype)
    return X, y, X_test


def saver(prediction):
    # MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
    assert prediction.ndim == 1
    assert prediction.shape[0] == 300

    # AND SAVE EXACTLY AS SHOWN BELOW
    np.save('prediction.npy', prediction)

if __name__ == '__main__':
    X, y, X_test = loader()
    X = np.rot90(np.moveaxis(X, 1, 3), 0, (1, 2))

    plt.imshow(X[2])
    plt.title(y[0])
    plt.show()

    # plt.imshow(X[:, 0, 0])
    # plt.title(y[0])
    # plt.show()
    #
    # plt.imshow(X[:, 0, 0])
    # plt.show()

