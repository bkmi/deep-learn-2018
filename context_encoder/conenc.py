import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_masked_dataset(data, labels, batch_size, image_shape=(28, 28), mask=None):
    def gen():
        for image, label in zip(data, labels):
            yield image, label
    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), (image_shape, ()))
    if mask is None:
        mask = np.ones(shape=image_shape)
    return ds.map(lambda x, y: (x * mask, y)).repeat().batch(batch_size)

# fix shape = 3 for a pic
def create_square_mask(image_shape=(28, 28, 1), blank_shape=(14, 14, 1)):
    for i, j in zip(image_shape, blank_shape):
        assert i >= j
    mask = np.ones(shape=image_shape)
    srow, scol = (int((x - y)/2) for x, y in zip(image_shape, blank_shape))
    mask[srow:srow + blank_shape[0], scol:scol + blank_shape[1]] = 0.0
    assert mask.shape == tuple(image_shape)
    return mask


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def main():
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_dataset = create_masked_dataset(
        x_train,
        y_train,
        batch_size=10,
        image_shape=x_train.shape[1:],
        mask=create_square_mask(image_shape=x_train.shape[1:], blank_shape=(16, 16, 3))
    )
    valid_dataset = create_masked_dataset(
        x_test,
        y_test,
        batch_size=20,
        image_shape=x_train.shape[1:],
        mask=create_square_mask(image_shape=x_train.shape[1:], blank_shape=(16, 16, 3))
    )

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    image, label = iterator.get_next()

    train_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = valid_dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_handle = sess.run(train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())

        train_img, train_label = sess.run([image, label], feed_dict={handle: train_handle})

        for i in train_img:
            plt.imshow(i)
            plt.show()

        valid_img = sess.run([image], feed_dict={handle: valid_handle})


if __name__ == '__main__':
    main()
