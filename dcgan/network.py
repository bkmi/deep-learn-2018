import tensorflow as tf
import functools

from pathlib import Path


def lazy_property(function):
    # https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class DCGAN:
    def __init__(self, real_batch, latent_batch, learning_rate=0.0002, beta1=0.5, log_bias=1e-12):
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.log_bias = log_bias

        self.real_batch = real_batch
        self.fake_batch = self._generator(latent_batch)

        self.d_real = self._discriminator(self.real_batch, reuse=False)
        self.d_fake = self._discriminator(self.fake_batch, reuse=True)

        self.d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(
            self._discriminator_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
            name='discriminator_solver'
        )
        self.g_opt = tf.train.AdamOptimizer().minimize(
            self._generator_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'),
            name='generator_solver'
        )

    def _generator(self, latent):
        NotImplementedError('Base class')

    def _discriminator(self, image, reuse=False):
        NotImplementedError('Base class')

    @lazy_property
    def _generator_loss(self):
        return -tf.reduce_mean(
            tf.log(self.d_fake + self.log_bias),
            name='generator_loss'
        )

    @lazy_property
    def _discriminator_loss(self):
        return -tf.reduce_mean(
            tf.log(self.d_real + self.log_bias) + tf.log(1. - self.d_fake + self.log_bias),
            name='discriminator_loss'
        )

    def train_batch(self, session, feed_dict):
        with tf.variable_scope(name_or_scope="training"):
            _, d_loss = session.run(
                [self.d_opt, self._discriminator_loss],
                feed_dict=feed_dict
            )
            _, g_loss = session.run(
                [self.g_opt, self._generator_loss],
                feed_dict=feed_dict
            )
        return d_loss, g_loss

    @lazy_property
    def _fake_grid(self):
        def image_grid(x, size=4):
            t = tf.unstack(x[:size * size], num=size * size, axis=0)
            rows = [tf.concat(t[i * size:(i + 1) * size], axis=0)
                    for i in range(size)]
            image = tf.concat(rows, axis=1)
            return image

        with tf.variable_scope(name_or_scope="generate_grid"):
            fake_batch_int = tf.image.convert_image_dtype(self.fake_batch, tf.uint8)
            fake_grid = image_grid(fake_batch_int, size=4)
        return fake_grid

    @lazy_property
    def _filename_grid(self):
        return tf.placeholder_with_default(str(Path(__file__, 'image.png')), shape=())

    @lazy_property
    def _write_grid(self):
        with tf.variable_scope(name_or_scope="write_grid"):
            png = tf.image.encode_png(self._fake_grid)
            write_file = tf.write_file(self._filename_grid, png)
        return write_file

    def generate(self, session):
        tiled = session.run(self._fake_grid, feed_dict={self.is_training: False})
        return tiled

    def write_grid(self, session, path=None):
        feed_dict = {self.is_training: False}
        if path is not None:
            feed_dict[self._filename_grid] = str(path)
        session.run([self._write_grid], feed_dict=feed_dict)
        return None


class MNISTDCGAN(DCGAN):
    def __init__(self, *args, **kwargs):
        self.filters_max = 64
        super().__init__(*args, **kwargs)

    def _generator(self, latent):
        with tf.variable_scope(name_or_scope="generator"):
            x = tf.layers.dense(latent, 3 * 3 * self.filters_max, activation=None)
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.reshape(x, shape=(-1, 3, 3, self.filters_max))  # 3, 3
            x = tf.layers.conv2d_transpose(x, filters=self.filters_max // 2, kernel_size=3, strides=2, activation=tf.nn.relu)  # 7, 7
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.layers.conv2d_transpose(x, filters=self.filters_max // 4, kernel_size=2, strides=2, activation=tf.nn.relu)  # 14, 14
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=2, strides=2, activation=tf.nn.tanh)  # 28, 28
        return x

    def _discriminator(self, image, reuse=False):
        conv_kwargs = {'kernel_size': 2, 'strides': 2, 'activation': tf.nn.leaky_relu, 'reuse': reuse}
        bnorm_kwargs = {'training': self.is_training, 'reuse': reuse}
        with tf.variable_scope(name_or_scope="discriminator", reuse=reuse):
            x = tf.layers.conv2d(image, filters=self.filters_max // 4, name='c0', **conv_kwargs)  # 14, 14
            x = tf.layers.batch_normalization(x, name='b0', **bnorm_kwargs)
            x = tf.layers.conv2d(x, filters=self.filters_max // 2, name='c1', **conv_kwargs)  # 7, 7
            x = tf.layers.batch_normalization(x, name='b1', **bnorm_kwargs)
            x = tf.layers.conv2d(x, filters=self.filters_max, name='c2', **conv_kwargs)  # 3, 3
            x = tf.layers.batch_normalization(x, name='b2', **bnorm_kwargs)
            x = tf.layers.flatten(x, name='flat')
            x = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='d0', reuse=reuse)
            # x = tf.layers.batch_normalization(x, name='b3', **bnorm_kwargs)
        return x


class CIFARDCGAN(DCGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generator(self, latent):
        with tf.variable_scope(name_or_scope="generator"):
            x = tf.layers.dense(latent, 4 * 4 * 512, activation=None)
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.reshape(x, shape=(-1, 4, 4, 512))  # 4, 4
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=2, activation=tf.nn.relu)  # 7, 7
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=2, strides=2, activation=tf.nn.relu)  # 14, 14
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=2, strides=2, activation=tf.nn.tanh)  # 28, 28
        return x

    def _discriminator(self, image, reuse=False):
        conv_kwargs = {'kernel_size': 2, 'strides': 2, 'activation': tf.nn.leaky_relu, 'reuse': reuse}
        bnorm_kwargs = {'training': self.is_training, 'reuse': reuse}
        with tf.variable_scope(name_or_scope="discriminator", reuse=reuse):
            x = tf.layers.conv2d(image, filters=128, name='c0', **conv_kwargs)  # 14, 14
            x = tf.layers.batch_normalization(x, name='b0', **bnorm_kwargs)
            x = tf.layers.conv2d(x, filters=256, name='c1', **conv_kwargs)  # 7, 7
            x = tf.layers.batch_normalization(x, name='b1', **bnorm_kwargs)
            x = tf.layers.conv2d(x, filters=512, name='c2', **conv_kwargs)  # 3, 3
            x = tf.layers.batch_normalization(x, name='b2', **bnorm_kwargs)
            x = tf.layers.flatten(x, name='flat')
            x = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='d0', reuse=reuse)
            x = tf.layers.batch_normalization(x, name='b3', **bnorm_kwargs)
        return x
