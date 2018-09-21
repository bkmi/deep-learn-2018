import tensorflow as tf


def squeeze_scale_labeled_data(data_label_tuple, scale_tanh=True, squeeze=True):
    x, y = data_label_tuple
    if scale_tanh:
        x = (2.0 * x / x.max()) - 1
    if squeeze:
        y = y.squeeze()
    return x, y


def create_keras_dataset(keras_tuple, scale_tanh=True, squeeze=True):
    (x_train, y_train), (x_test, y_test) = [
        squeeze_scale_labeled_data(data_label, scale_tanh=scale_tanh, squeeze=squeeze)
        for data_label
        in keras_tuple
    ]
    return (x_train, y_train), (x_test, y_test)


def create_cifar10(scale_tanh=True, squeeze=True):
    return create_keras_dataset(tf.keras.datasets.cifar10.load_data(), scale_tanh=scale_tanh, squeeze=squeeze)


def create_mnist(scale_tanh=True, squeeze=True):
    return create_keras_dataset(tf.keras.datasets.mnist.load_data(), scale_tanh=scale_tanh, squeeze=squeeze)


def create_fashion_mnist(scale_tanh=True, squeeze=True):
    return create_keras_dataset(tf.keras.datasets.fashion_mnist.load_data(), scale_tanh=scale_tanh, squeeze=squeeze)


def create_dataset(images, labels, batch_size, buffer_size=10000, repeat=False):
    def gen():
        for image, label in zip(images, labels):
            yield image, label
    ds = tf.data.Dataset.from_generator(
        gen,
        (tf.float32, tf.int32),
        (images.shape[1:], labels.shape[1:])
    )
    if repeat:
        return ds.map(lambda x, y: (x, y)).repeat().shuffle(buffer_size=buffer_size).batch(batch_size)
    else:
        return ds.map(lambda x, y: (x, y)).shuffle(buffer_size=buffer_size).batch(batch_size)
