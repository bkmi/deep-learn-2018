import tensorflow as tf


def create_dataset(data, labels, batch_size, train_shape=(28, 28)):
    def gen():
        for image, label in zip(data, labels):
            yield image, label
    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), (train_shape, ()))
    return ds.repeat().batch(batch_size)


if __name__ == '__main__':
    # https://stackoverflow.com/questions/50666681/how-to-load-mnist-via-tensorflow-including-download/50669146
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_dataset = create_dataset(x_train, y_train, 10)
    valid_dataset = create_dataset(x_test, y_test, 20)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    image, label = iterator.get_next()

    train_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = valid_dataset.make_one_shot_iterator()

    # A toy network
    y = tf.layers.dense(tf.layers.flatten(image), 1, activation=tf.nn.relu)
    loss = tf.losses.mean_squared_error(tf.squeeze(y), label)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        train_handle = sess.run(train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())

        # Run training
        train_loss, train_img, train_label = sess.run([loss, image, label],
                                                      feed_dict={handle: train_handle})
        # train_image.shape = (10, 784)

        # Run validation
        valid_pred, valid_img = sess.run([y, image],
                                         feed_dict={handle: valid_handle})
        # test_image.shape = (20, 784)
