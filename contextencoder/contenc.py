import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial


def valid_conv_math(input_width, filter_width, stride):
    return (input_width - filter_width)//stride + 1


def valid_conv_transpose_math(input_width, filter_width, stride):
    return (input_width - 1) * stride + filter_width


def image_grid(x, size=4):
    t = tf.unstack(x[:size * size], num=size * size, axis=0)
    rows = [tf.concat(t[i * size:(i + 1) * size], axis=0)
            for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image


def create_cifar10(normalize=True, squeeze=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # if normalize:
    #     x_train, x_test = (i / x_train.max() for i in (x_train, x_test))
    if squeeze:
        y_train, y_test = (i.squeeze() for i in (y_train, y_test))
    return (x_train, y_train), (x_test, y_test)


def create_dataset(data, labels, batch_size, image_shape=(28, 28, 1), label_shape=(), buffer_size=10000, repeat=False):
    def gen():
        for image, label in zip(data, labels):
            yield image, label
    ds = tf.data.Dataset.from_generator(
        gen,
        (tf.float32, tf.int32),
        (image_shape, label_shape)
    )
    if repeat:
        return ds.map(lambda x, y: (x, y)).repeat().shuffle(buffer_size=buffer_size).batch(batch_size)
    else:
        return ds.map(lambda x, y: (x, y)).shuffle(buffer_size=buffer_size).batch(batch_size)


class ContextEncoder:
    def __init__(self, batch_tensor, filters=48, count_conv=4, learning_rate=1e-3):
        self.image_batch = batch_tensor / tf.reduce_max(batch_tensor)
        self.whitened_image_batch = tf.map_fn(lambda im: tf.image.per_image_standardization(im), self.image_batch)
        self.image_dim = [int(i) for i in self.whitened_image_batch.shape[1:]]
        self.mask_dim = [self.image_dim[0] // 4, self.image_dim[1] // 4, 3]
        self.border_px = 2
        self.learning_rate = learning_rate

        with tf.variable_scope('mask'):
            self.mask = tf.image.resize_image_with_crop_or_pad(
                tf.ones(shape=self.mask_dim, dtype=tf.bool),
                self.image_dim[0],
                self.image_dim[1]
            )
        with tf.variable_scope('overlap_region'):
            self.over = tf.image.resize_image_with_crop_or_pad(
                tf.ones(shape=[self.mask_dim[0] + self.border_px, self.mask_dim[1] + self.border_px, 3], dtype=tf.bool),
                self.image_dim[0],
                self.image_dim[1]
            )
        with tf.variable_scope('border_region'):
            self.border = tf.logical_xor(self.mask, self.over)

        with tf.variable_scope('whitened_masked_input'):
            self.whitened_masked_batch = tf.add(
                tf.cast(~self.mask, dtype=tf.float32) * self.whitened_image_batch,
                tf.reduce_mean(self.whitened_image_batch, axis=[1, 2], keepdims=True) * tf.cast(self.mask, dtype=tf.float32)
            )
        with tf.variable_scope('masked_input'):
            self.masked_batch = tf.add(
                tf.cast(~self.mask, dtype=tf.float32) * self.image_batch,
                tf.reduce_mean(self.whitened_image_batch, axis=[1, 2], keepdims=True) * tf.cast(self.mask, dtype=tf.float32)
            )
        with tf.variable_scope('cutout'):
            self.cutout_batch = tf.reshape(
                tf.boolean_mask(self.whitened_image_batch, self.mask, axis=1), shape=[-1] + self.mask_dim
            )

        self.is_training = tf.placeholder_with_default(True, shape=(), name='is_training')
        self.count_conv = count_conv - 1
        self.filters = filters

        def calculate_reduced_dim(image_dim):
            conv = partial(valid_conv_math, filter_width=3, stride=2)
            reduced_dim = valid_conv_math(image_dim, filter_width=3, stride=1)
            for _ in range(self.count_conv):
                reduced_dim = conv(reduced_dim)
            return reduced_dim

        self.convd_dim = calculate_reduced_dim(self.image_dim[0])
        self.latent_dim = self.filters * self.convd_dim ** 2

        self.z = self._encoder()
        self.decoded = self._decoder()
        with tf.variable_scope('padded_decoded_including_overlap'):
            self.padded_decoded = tf.image.resize_image_with_crop_or_pad(
                self.decoded,
                int(self.whitened_image_batch.shape[1]),
                int(self.whitened_image_batch.shape[2])
            )
        with tf.variable_scope('output_batch'):
            self.output_batch = tf.reshape(
                tf.boolean_mask(self.padded_decoded, self.mask, axis=1),
                shape=[-1] + self.mask_dim,
                name='output_batch'
            )
        with tf.variable_scope('output_padded'):
            self.output_padded = tf.image.resize_image_with_crop_or_pad(
                self.output_batch,
                self.image_dim[0],
                self.image_dim[1]
            )

        self.d_fake = self._discriminator(self.output_batch, reuse=False)
        self.d_real = self._discriminator(self.cutout_batch, reuse=True)

        self.loss, self.optimization, self.reconstruction_loss, self.adversarial_loss = self._context_encoder_loss()
        self.d_loss, self.d_opt = self._discriminator_loss()

    def _encoder(self):
        with tf.variable_scope('context_encoder'):
            conv_kwargs = {'kernel_size': 3, 'padding': 'valid', 'activation': tf.nn.leaky_relu}
            count_filters = self.filters // (2 ** (self.count_conv - 1))
            x = tf.layers.conv2d(self.whitened_masked_batch, strides=1, filters=count_filters, **conv_kwargs)
            x = tf.layers.batch_normalization(x, training=self.is_training)
            for i in range(self.count_conv):
                count_filters = self.filters // (2 ** (self.count_conv - i - 1))
                x = tf.layers.conv2d(x, strides=2, filters=count_filters, **conv_kwargs)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, units=self.latent_dim, activation=tf.nn.leaky_relu)
        return x

    def _decoder(self):
        with tf.variable_scope('context_decoder'):
            conv_kwargs = {'kernel_size': 3, 'strides': 1, 'padding': 'valid', 'activation': tf.nn.relu}
            x = tf.layers.dense(self.z, units=self.latent_dim, activation=tf.nn.relu)
            x = tf.reshape(x, shape=[-1, self.convd_dim, self.convd_dim, self.filters])
            for i in range(self.count_conv + 1):
                count_filters = self.filters // (2 ** i)
                x = tf.layers.conv2d_transpose(x, filters=count_filters, **conv_kwargs)
            decoded = tf.layers.conv2d(x, kernel_size=3, filters=3, padding='same', activation=tf.nn.sigmoid)
        return decoded

    def _discriminator(self, input_tensor, reuse=False):
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
            x = tf.layers.batch_normalization(x, training=self.is_training)
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
        with tf.variable_scope(name_or_scope="context_encoder_loss"):
            reconstruction_loss = tf.reduce_sum(
                tf.squared_difference(
                    tf.boolean_mask(self.image_batch, self.mask, axis=1),
                    tf.boolean_mask(self.padded_decoded, self.mask, axis=1)
                ),
                axis=[1],
                name='reconstruction_loss_cutout'
            )

            reconstruction_loss += 10 * tf.reduce_sum(
                tf.squared_difference(
                    tf.boolean_mask(self.image_batch, self.border, axis=1),
                    tf.boolean_mask(self.padded_decoded, self.border, axis=1)
                ),
                axis=[1],
                name='reconstruction_loss_overlap'
            )

            reconstruction_loss = tf.reduce_mean(reconstruction_loss, name='reconstruction_loss_total')

            adversarial_loss = -tf.reduce_mean(
                tf.log(self.d_fake + 1e-12),
                name='adversarial_loss'
            )

            loss = reconstruct_coef * reconstruction_loss + (1 - reconstruct_coef) * adversarial_loss

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='context_encoder')
            var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='context_decoder')
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                loss,
                var_list=var_list,
                name='context_encoder_solver'
            )
        return loss, opt, reconstruction_loss, adversarial_loss

    def _discriminator_loss(self):
        with tf.variable_scope(name_or_scope="discriminator_loss"):
            discriminator_loss = -tf.reduce_mean(
                tf.log(self.d_real + 1e-12) + tf.log(1. - self.d_fake + 1e-12),
                name='discriminator_loss'
            )

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate / 10).minimize(
                discriminator_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
                name='discriminator_solver'
            )
        return discriminator_loss, opt

    def train_batch(self, session, feed_dict):
        with tf.variable_scope(name_or_scope="training"):
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
        with tf.variable_scope(name_or_scope="compute"):
            feed_dict[self.is_training] = False
            image, masked, decoded = session.run(
                [self.whitened_image_batch, self.whitened_masked_batch, self.output_padded],
                feed_dict=feed_dict
            )
        return image, masked, decoded

    def print_images(self, session, feed_dict, directory):
        with tf.device('/cpu:0'):
            with tf.variable_scope(name_or_scope="print_images"):
                feed_dict[self.is_training] = False

                combined_images = self.masked_batch + self.output_padded
                combined_images = tf.image.convert_image_dtype(combined_images, tf.uint8)
                tiled_images = image_grid(combined_images, size=4)
                png = tf.image.encode_png(tiled_images)

                def name_new_tiled_image(direct):
                    direct = Path(direct)
                    direct.mkdir(exist_ok=True)
                    contents = sorted(direct.glob('epoch_*.png'))
                    try:
                        current_count = contents[-1].name.split(sep='_')[-1].split(sep='.')[0]
                    except IndexError:
                        current_count = -1

                    if int(current_count) < 9:
                        return '000' + str(int(current_count) + 1)
                    elif int(current_count) < 99:
                        return '00' + str(int(current_count) + 1)
                    elif int(current_count) < 999:
                        return '0' + str(int(current_count) + 1)
                    else:
                        return str(int(current_count) + 1)
                write = tf.write_file(str(Path(directory, 'epoch_' + name_new_tiled_image(directory) + '.png')), png)

                tiled, _ = session.run([tiled_images, write], feed_dict=feed_dict)
            return tiled


def main(tensorboard=False, save_directory=False):
    (x_train, y_train), (x_test, y_test) = create_cifar10()
    batch_size = 100
    epochs = 2000

    train_dataset, valid_dataset = [
        create_dataset(
            x,
            y,
            batch_size=z,
            image_shape=x.shape[1:],
            repeat=w
        ) for x, y, z, w in zip((x_train, x_test), (y_train, y_test), (batch_size, 25), (False, True))
    ]

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    image, label = iterator.get_next()

    ce = ContextEncoder(image)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if tensorboard:
            writer = tf.summary.FileWriter('/tmp/contextencoder/1')
            writer.add_graph(sess.graph)

        if save_directory:
            save_directory = Path(save_directory)
            save_directory.mkdir(exist_ok=True)
            images_dir = Path(save_directory, 'images')
            images_dir.mkdir(exist_ok=True)
            weight_dir = Path(save_directory, 'models')
            weight_dir.mkdir(exist_ok=True)

        valid_iterator = valid_dataset.make_one_shot_iterator()
        valid_handle = sess.run(valid_iterator.string_handle())

        for ep in range(epochs):
            train_iterator = train_dataset.make_one_shot_iterator()
            train_handle = sess.run(train_iterator.string_handle())
            while True:
                try:
                    ce_loss, r_loss, a_loss, d_loss = ce.train_batch(sess, feed_dict={handle: train_handle})
                except tf.errors.OutOfRangeError:
                    break
            print(f'ce: {ce_loss}, rec: {r_loss}, adv: {a_loss}, dis: {d_loss}')

            # _, masked, decoded = ce.compute_batch(sess, feed_dict={handle: valid_handle})
            if save_directory and (ep % 10 == 0):
                _ = ce.print_images(sess, feed_dict={handle: valid_handle}, directory=images_dir)
                saver.save(sess, str(Path(weight_dir, 'save.ckpt')))


if __name__ == '__main__':
    main(tensorboard=False, save_directory=Path(Path(__file__).parent, 'save_here'))
