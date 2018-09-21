import tensorflow as tf

from dcgan.data import create_mnist, create_dataset


(x_train, y_train), (x_test, y_test) = create_mnist()
batch_size = 128
epochs = 2000

train_dataset, valid_dataset = [
    create_dataset(
        x,
        y,
        batch_size=z,
        repeat=w
    ) for x, y, z, w in zip((x_train, x_test), (y_train, y_test), (batch_size, 25), (False, True))
]