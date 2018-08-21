import tensorflow as tf
import numpy as np
import bbox
import matplotlib.pyplot as plt

from functools import partial


def valid_conv_math(input_width, filter_width, stride):
    return (input_width - filter_width)//stride + 1


def valid_conv_transpose_math(input_width, filter_width, stride):
    return (input_width - 1) * stride + filter_width


class ContextEncoder:
    def __init__(self, batch_tensor, mask, batch_size=100, image_dim=32, filters=48, count_conv=4):
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
        self.image_batch = batch_tensor
        self.mask = mask
        mask_length = int(np.sqrt(np.sum(self.mask) // 3))
        self.mask_dim = [-1, mask_length, mask_length, 3]
        self.masked_batch = (~self.mask) * self.image_batch
        self.cutout = tf.reshape(tf.boolean_mask(self.image_batch, self.mask, axis=1),
                                 shape=self.mask_dim)

        self.z = self._encoder()
        self.decoded = self._decoder()
        self.padded_decoded = tf.image.resize_image_with_crop_or_pad(
            self.decoded,
            int(self.image_batch.shape[1]),
            int(self.image_batch.shape[2])
        )

        self.d_fake = self._discriminator(self.decoded, reuse=False, is_training=False)
        self.d_real = self._discriminator(self.cutout, reuse=True, is_training=True)

        self.loss, self.optimization, self.reconstruction_loss, self.adversarial_loss = self._context_encoder_loss()
        self.d_loss, self.d_opt = self._discriminator_loss()

    def _encoder(self):
        with tf.variable_scope('context'):
            with tf.variable_scope('encoder'):
                conv_kwargs = {'kernel_size': 3, 'padding': 'valid', 'activation': tf.nn.leaky_relu}
                count_filters = self.filters // (2 ** (self.count_conv - 1))
                x = tf.layers.conv2d(self.masked_batch, strides=1, filters=count_filters, **conv_kwargs)
                x = tf.layers.batch_normalization(x, training=self.is_training)
                for i in range(self.count_conv):
                    count_filters = self.filters // (2 ** (self.count_conv - i - 1))
                    x = tf.layers.conv2d(x, strides=2, filters=count_filters, **conv_kwargs)
                x = tf.layers.flatten(x)
                x = tf.layers.dense(x, units=self.latent_dim, activation=tf.nn.leaky_relu)
        return x

    def _decoder(self):
        with tf.variable_scope('context'):
            with tf.variable_scope('decoder'):
                conv_kwargs = {'kernel_size': 3, 'strides': 1, 'padding': 'valid', 'activation': tf.nn.relu}
                x = tf.layers.dense(self.z, units=self.latent_dim, activation=tf.nn.relu)
                x = tf.reshape(x, shape=[-1, self.convd_dim, self.convd_dim, self.filters])
                for i in range(self.count_conv):
                    count_filters = self.filters // (2 ** i)
                    x = tf.layers.conv2d_transpose(x, filters=count_filters, **conv_kwargs)
                decoded = tf.layers.conv2d(x, kernel_size=3, filters=3, padding='same', activation=tf.nn.sigmoid)
        return decoded

    def _discriminator(self, input_tensor, reuse=False, is_training=False):
        conv_kwargs = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'valid',
            'activation': tf.nn.leaky_relu,
            'reuse': reuse
        }
        count_filters = self.filters // (2 ** (self.count_conv - 1))
        with tf.variable_scope(name_or_scope="discriminator", reuse=reuse):
            x = tf.layers.conv2d(input_tensor, filters=count_filters, name='conv0', **conv_kwargs)
            x = tf.layers.batch_normalization(x, training=is_training)
            for i in range(self.count_conv - 1):
                conv_name = 'conv' + str(i + 1)
                count_filters = self.filters // (2 ** (self.count_conv - i - 1))
                x = tf.layers.conv2d(x, filters=count_filters, name=conv_name, **conv_kwargs)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(
                x,
                units=1,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                activation=tf.nn.sigmoid,
                reuse=reuse
            )
        return x

    def _context_encoder_loss(self, reconstruct_coef=0.999):
        # mask is 1s when dropped, 0s when not dropped
        reconstruction_loss = tf.reduce_sum(
            tf.squared_difference(self.cutout, self.decoded),
            axis=[1]
        )
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        adversarial_loss = -tf.reduce_mean(
            tf.log(self.d_fake + 1e-12),
            name='generator_loss'
        )

        loss = reconstruct_coef * reconstruction_loss + (1 - reconstruct_coef) * adversarial_loss

        opt = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(
            loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='context'),
            name='context_encoder_solver'
        )
        return loss, opt, reconstruction_loss, adversarial_loss

    def _discriminator_loss(self):
        discriminator_loss = -tf.reduce_mean(
            tf.log(self.d_real + 1e-12) + tf.log(1. - self.d_fake + 1e-12),
            name='discriminator_loss'
        )

        opt = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(
            discriminator_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
            name='discriminator_solver'
        )
        return discriminator_loss, opt

    def train_batch(self, session, feed_dict):
        _, d_loss = session.run(
            [self.d_opt, self.d_loss],
            feed_dict=feed_dict
        )
        _, ce_loss, r_loss, a_loss = session.run(
            [self.optimization, self.loss, self.reconstruction_loss, self.adversarial_loss],
            feed_dict=feed_dict
        )
        return ce_loss, r_loss, a_loss, d_loss

    def compute_batch(self, session, feed_dict):
        feed_dict[self.is_training] = False
        image, masked, decoded = session.run(
            [self.image_batch, self.masked_batch, self.padded_decoded],
            feed_dict=feed_dict
        )
        return image, masked, decoded


def main():
    (x_train, y_train), (x_test, y_test) = bbox.create_cifar10()
    blank_dim = [i//4 for i in x_train.shape[1:3]]
    batch_size = 500
    epochs = 2

    mask = bbox.create_square_mask(image_shape=x_train.shape[1:], blank_dim=blank_dim)

    train_dataset, valid_dataset = [
        bbox.create_dataset(
            x,
            y,
            batch_size=batch_size,
            image_shape=x.shape[1:],
        ) for x, y in zip((x_train, x_test), (y_train, y_test))
    ]

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    image, label = iterator.get_next()

    ce = ContextEncoder(image, mask)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(epochs):
            train_iterator = train_dataset.make_one_shot_iterator()
            train_handle = sess.run(train_iterator.string_handle())
            while True:
                try:
                    ce_loss, r_loss, a_loss, d_loss = ce.train_batch(sess, feed_dict={handle: train_handle})
                except tf.errors.OutOfRangeError:
                    break
            print(f'ce: {ce_loss}, rec: {r_loss}, adv: {a_loss}, dis: {d_loss}')

        valid_iterator = valid_dataset.make_one_shot_iterator()
        valid_handle = sess.run(valid_iterator.string_handle())

        image, masked, decoded = ce.compute_batch(sess, feed_dict={handle: valid_handle})
        plt.imshow(masked[0])
        plt.show()
        plt.imshow(masked[0] + decoded[0])
        plt.show()


if __name__ == '__main__':
    main()
