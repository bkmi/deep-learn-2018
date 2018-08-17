import tensorflow as tf
import numpy as np
import bbox

from functools import partial


def valid_conv_math(input_width, filter_width, stride):
    return (input_width - filter_width)//stride + 1


def valid_conv_transpose_math(input_width, filter_width, stride):
    return (input_width - 1) * stride + filter_width


class ContextEncoder:
    def __init__(self, input_tensor, batch_size=100, image_dim=32, filters=48, count_conv=4):
        self.batch_size = tf.placeholder_with_default(np.int64(batch_size), shape=())

        self.count_conv = count_conv - 1
        self.filters = filters
        conv = partial(valid_conv_math, filter_width=3, stride=2)
        reduced_dim = valid_conv_math(image_dim, filter_width=3, stride=1)
        for _ in range(self.count_conv):
            reduced_dim = conv(reduced_dim)
        self.convd_dim = reduced_dim
        self.latent_dim = self.filters * self.convd_dim ** 2

        self.is_training = tf.placeholder_with_default(True, shape=())
        self.image_batch = input_tensor

        self.z = self._encoder()
        self.decoded = self._decoder()

        self.loss, self.optimization, self.reconstruction_loss, self.latent_loss = self._make_loss_opt()

    def _encoder(self):
        conv_kwargs = {'kernel_size': 3, 'filters': self.filters, 'padding': 'valid', 'activation': tf.nn.leaky_relu}
        x = tf.layers.conv2d(self.image_batch, strides=1, **conv_kwargs)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        for _ in range(self.count_conv):
            x = tf.layers.conv2d(x, strides=2, **conv_kwargs)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=self.latent_dim, activation=tf.nn.leaky_relu)
        return x

    def _decoder(self):
        conv_kwargs = {'filters': self.filters, 'padding': 'valid', 'activation': tf.nn.leaky_relu}
        x = tf.layers.dense(self.z, units=self.latent_dim, activation=tf.nn.leaky_relu)
        x = tf.reshape(x, shape=[-1, self.convd_dim, self.convd_dim, self.filters])
        for _ in range(self.count_conv):
            x = tf.layers.conv2d_transpose(x, kernel_size=4, strides=2, **conv_kwargs)
        x = tf.layers.conv2d_transpose(x, kernel_size=3, strides=1, **conv_kwargs)
        x = tf.layers.conv2d(x, kernel_size=3, filters=3, padding='same', activation=tf.nn.leaky_relu)
        decoded = tf.layers.conv2d(x, kernel_size=3, filters=3, padding='same', activation=tf.nn.sigmoid)
        return decoded

    def _make_loss_opt(self):
        reconstruction_loss = tf.reduce_sum(self.image_batch * tf.log(1e-10 + self.decoded) +
                                            (1 - self.image_batch) * tf.log(1e-10 + 1 - self.decoded),
                                            axis=[1, 2, 3])
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        latent_loss = 0.5 * tf.reduce_sum(1 + self.z_log_var - self.z_mean ** 2 - tf.exp(self.z_log_var), axis=1)
        latent_loss = tf.reduce_mean(latent_loss)
        loss = -(reconstruction_loss + latent_loss)

        opt = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
        return loss, opt, reconstruction_loss, latent_loss

    def train(self, session, images):
        session.run(self.iterator.initializer, feed_dict={self.image_input: images})

        while True:
            try:
                _, loss, reconstruction_loss, latent_loss, decoded = session.run(
                    [self.optimization, self.loss, self.reconstruction_loss, self.latent_loss, self.decoded],
                    feed_dict={self.is_training: True}
                )
            except tf.errors.OutOfRangeError:
                break

        return loss, reconstruction_loss, latent_loss, decoded


def main():
    (x_train, y_train), (x_test, y_test) = bbox.create_cifar10()
    blank_dim = [i//4 for i in x_train.shape[1:3]]
    batch_size = 35

    train_dataset, valid_dataset = [
        bbox.create_masked_dataset(
            x,
            y,
            batch_size=batch_size,
            image_shape=x.shape[1:],
            mask=bbox.create_square_mask(image_shape=x.shape[1:], blank_dim=blank_dim)
        ) for x, y in zip((x_train, x_test), (y_train, y_test))
    ]

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    image, cutout, label = iterator.get_next()

    train_iterator, valid_iterator = [i.make_one_shot_iterator() for i in (train_dataset, valid_dataset)]
    ce = ContextEncoder(image)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_handle, valid_handle = sess.run([train_iterator.string_handle(), valid_iterator.string_handle()])
        a, b = sess.run([image, ce.decoded], feed_dict={handle: train_handle})


        train_img, train_cutout, train_label = sess.run([image, cutout, label], feed_dict={handle: train_handle})
        valid_img, valid_cutout, valid_label = sess.run([image, cutout, label], feed_dict={handle: valid_handle})


if __name__ == '__main__':
    main()
