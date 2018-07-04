import numpy as np

from sklearn.preprocessing import LabelBinarizer


def vectorize_sequence(sequence):
    lb = LabelBinarizer()
    lb.fit(list(set(sequence)))
    sequence = list(sequence)
    return lb.transform(sequence)


def load(verbose=False, vectorize=True):
    with np.load('rnn-challenge-data.npz') as fh:
        data_x, data_y = fh['data_x'], fh['data_y']
        vali_x, vali_y = fh['val_x'], fh['val_y']
        test_x = fh['test_x']

    if vectorize:
        data_x, vali_x, test_x = map(lambda x: np.asarray([vectorize_sequence(seq) for seq in x]),
                                     [data_x, vali_x, test_x])
        data_y, vali_y = map(lambda seq: np.asarray(vectorize_sequence(seq)),
                             [data_y, vali_y])

    if verbose:
        print('name_x: (count_sequences, len_sequences, count_vocab), dtype')
        for name, data in zip(('data_x', 'vali_x', 'text_x'), (data_x, vali_x, test_x)):
            print(f'{name}: {data.shape}, {data.dtype}')

        print('name_y: (count_sequences, count_possible_labels), dtype')
        for name, data in zip(('data_y', 'vali_y'), (data_y, vali_y)):
            print(f'{name}: {data.shape}, {data.dtype}')

    return data_x, data_y, vali_x, vali_y, test_x
