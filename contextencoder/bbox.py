import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_masked_dataset(data, labels, batch_size, image_shape=(28, 28, 1), label_shape=(), mask=None):
    def gen():
        for image, label in zip(data, labels):
            yield image, image, label
    ds = tf.data.Dataset.from_generator(
        gen,
        (tf.float32, tf.float32, tf.int32),
        (image_shape, image_shape, label_shape)
    )
    if mask is None:
        mask = np.ones(shape=image_shape, dtype=np.bool)
    return ds.map(lambda x, y, z: (x * mask, tf.boolean_mask(y, ~mask), z)).repeat().batch(batch_size)


def create_square_mask(image_shape=(28, 28, 1), blank_dim=(14, 14), negate=False):
    image_dim, count_chan = image_shape[:2], image_shape[2]
    for i, j in zip(image_dim, blank_dim):
        assert i >= j
    mask = np.ones(shape=image_shape, dtype=np.bool)
    srow, scol = (int((x - y)/2) for x, y in zip(image_dim, blank_dim))
    mask[srow:srow + blank_dim[0], scol:scol + blank_dim[1], :] = 0.0

    assert mask.shape == tuple(image_shape)
    if negate:
        return ~mask
    else:
        return mask
    

def create_cifar10(normalize=True, squeeze=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if normalize:
        x_train, x_test = (i / x_train.max() for i in (x_train, x_test))
    if squeeze:
        y_train, y_test = (i.squeeze() for i in (y_train, y_test))
    return (x_train, y_train), (x_test, y_test)



def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def main():
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = (i/x_train.max() for i in (x_train, x_test))
    y_train, y_test = (i.squeeze() for i in (y_train, y_test))

    blank_dim = (i//4 for i in x_train.shape[1:3])
    train_dataset = create_masked_dataset(
        x_train,
        y_train,
        batch_size=10,
        image_shape=x_train.shape[1:],
        mask=create_square_mask(image_shape=x_train.shape[1:], blank_dim=blank_dim)
    )
    valid_dataset = create_masked_dataset(
        x_test,
        y_test,
        batch_size=20,
        image_shape=x_test.shape[1:],
        mask=create_square_mask(image_shape=x_test.shape[1:], blank_dim=blank_dim)
    )

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    image, cutout, label = iterator.get_next()

    train_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = valid_dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_handle = sess.run(train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())

        train_img, train_cutout, train_label = sess.run([image, cutout, label], feed_dict={handle: train_handle})

        for i in train_img:
            plt.imshow(i)
            plt.show()

        valid_img, valid_cutout, valid_label = sess.run([image, cutout, label], feed_dict={handle: valid_handle})


if __name__ == '__main__':
    main()
