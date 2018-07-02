import numpy as np

from sklearn.preprocessing import LabelBinarizer


lb = LabelBinarizer()
lb.fit(['C', 'A', 'G', 'T'])


def vectorize_sequence(sequence):
    sequence = list(sequence)
    return lb.transform(sequence)


def load(verbose=False, vectorize=True):
    with np.load('rnn-challenge-data.npz') as fh:
        data_x, data_y = fh['data_x'], fh['data_y']
        vali_x, vali_y = fh['val_x'], fh['val_y']
        test_x = fh['test_x']

    if vectorize:
        data_x = np.asarray([vectorize_sequence(seq) for seq in data_x])
        vali_x = np.asarray([vectorize_sequence(seq) for seq in vali_x])
        test_x = np.asarray([vectorize_sequence(seq) for seq in test_x])

    if verbose:
        print('name_x: datapts, sequences, dtype')
        print('data_x: {}, {}'.format(data_x.shape, data_x.dtype))
        print('vali_x: {}, {}'.format(vali_x.shape, vali_x.dtype))
        print('text_x: {}, {}'.format(test_x.shape, test_x.dtype))

    return data_x, data_y, vali_x, vali_y, test_x
