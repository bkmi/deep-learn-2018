import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import LabelBinarizer


def load_mnist_data():
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    test_x, test_y = test

    train_x, test_x = map(lambda x: x[..., None], [train_x, test_x])
    train_y, test_y = map(lambda x: LabelBinarizer().fit_transform(x), [train_y, test_y])

    return train_x, train_y, test_x, test_y


def img_show(img):
    plt.axis('off')
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.show()


# def generator(noise, is_training, weight_decay=2.5e-5):
#     with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
#         with tf.contrib.framework.arg_scope(
#                 [layers.fully_connected, layers.conv2d_transpose],
#                 activation_fn=tf.nn.relu,
#                 normalizer_fn=layers.batch_norm,
#                 weights_regularizer=layers.l2_regularizer(weight_decay)), \
#              tf.contrib.framework.arg_scope(
#                  [layers.batch_norm],
#                  is_training=is_training,
#                  zero_debias_moving_mean=True):
#             net = layers.fully_connected(noise, 1024)
#             net = layers.fully_connected(net, 7 * 7 * 128)
#             net = tf.reshape(net, [-1, 7, 7, 128])
#             net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
#             net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
#         net = layers.conv2d(
#             net,
#             1,
#             [4, 4],
#             normalizer_fn=None,
#             activation_fn=tf.sigmoid
#         )
#
#     return net

def generator(noise, is_training):
    with tf.variable_scope(name_or_scope="generator", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(noise, 128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.dense(x, 28 * 28, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
    return x


# def discriminator(img, weight_decay=4e-5):
#     with tf.variable_scope(name_or_scope=None, default_name="discriminator"):
#         with tf.contrib.framework.arg_scope([layers.conv2d, layers.fully_connected],
#                                             activation_fn=tf.nn.leaky_relu,
#                                             normalizer_fn=None,
#                                             weights_regularizer=layers.l2_regularizer(weight_decay),
#                                             biases_regularizer=layers.l2_regularizer(weight_decay)):
#             net = layers.conv2d(img, 64, [4, 4], stride=2)
#             net = layers.conv2d(net, 128, [4, 4], stride=2)
#             net = layers.flatten(net)
#             net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
#         net = layers.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)
#
#     return net

def discriminator(img):
    with tf.variable_scope(name_or_scope="discriminator", reuse=tf.AUTO_REUSE):
        x = tf.layers.flatten(img)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x


# def mix_real_and_fabricated(real_images, fabri_images):
#     dataset = tf.data.Dataset.from_tensor_slices(real_images)
#     dataset = dataset.shuffle(buffer_size=train_x.shape[0])
#     dataset = dataset.batch(batch_size=batch_size // 2)
#
#     iterator = dataset.make_initializable_iterator()
#     next_element = iterator.get_next()
#
#     labeled_images = (tf.concat([tf.cast(next_element, fabri_images.dtype), fabri_images], axis=0),
#                       tf.concat([tf.ones(batch_size // 2), tf.zeros(batch_size // 2)], axis=0))
#     labeled_images = tuple(map(lambda x: tf.random_shuffle(x, seed=42), labeled_images))
#     return labeled_images, iterator


# def pair_real_and_fabricated(real_images, fabri_images):
#     dataset = tf.data.Dataset.from_tensor_slices(real_images)
#     dataset = dataset.shuffle(buffer_size=train_x.shape[0])
#     dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=batch_size))
#
#     iterator = dataset.make_initializable_iterator()
#     next_element = iterator.get_next()
#
#     # paired_images = (tf.concat([tf.cast(next_element, fabri_images.dtype), fabri_images], axis=0),
#     #                  tf.concat([tf.ones(batch_size), tf.zeros(batch_size)], axis=0))
#     paired_images = (next_element, fabri_images)
#     # maybe it doesn't matter which order they are given in since it is a batch
#     # paired_images = tuple(map(lambda x: tf.random_shuffle(x, seed=42), paired_images))
#     return paired_images, iterator


def create_dataset(real_images):
    dataset = tf.data.Dataset.from_tensor_slices(real_images)
    dataset = dataset.shuffle(buffer_size=train_x.shape[0])
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=batch_size))

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return next_element, iterator


def iterate(predicate, images, iterator, session):
    session.run(iterator.initializer, feed_dict={real_images: images})
    while True:
        try:
            result = session.run(predicate)
        except tf.errors.OutOfRangeError:
            break
    return result


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


train_x, train_y, test_x, test_y = load_mnist_data()
batch_size, noise_dims = 32, 100
epochs = 1

real_images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
is_training = tf.placeholder_with_default(True, shape=())
# fabri_images = generator(tf.random_normal([batch_size, noise_dims]), is_training)
fabri_batch = generator(tf.random_uniform(shape=[batch_size, noise_dims], minval=-1, maxval=1),
                        is_training)
# paired_images, iterator = pair_real_and_fabricated(real_images, fabri_images)
real_batch, iterator = create_dataset(real_images)

full_batch = tf.concat([fabri_batch, real_batch], axis=0)
D_full = discriminator(full_batch)
D_real = D_full[batch_size:, ...]
D_fake = D_full[:batch_size, ...]

# D_real = discriminator(real_batch)
# D_fake = discriminator(fabri_batch)

D_loss = -tf.reduce_mean(tf.log(D_real + 1e-12) + tf.log(1. - D_fake + 1e-12))
G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-12))

# Now I need to keep track of which variables to update.
# start with this easy example then later implement the better loss function
# https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
# better loss: https://www.alexirpan.com/2017/02/22/wasserstein-gan.html
D_solver = tf.train.AdamOptimizer().minimize(
    D_loss,
    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
G_solver = tf.train.AdamOptimizer().minimize(
    G_loss,
    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(epochs):
        # images, labels = iterate(paired_images, train_x, iterator, sess)
        # images, labels = iterate(paired_images, test_x, iterator, sess)

        sess.run(iterator.initializer, feed_dict={real_images: train_x})
        while True:
            try:
                _, dis_loss = sess.run([D_solver, D_loss])
                _, gen_loss = sess.run([G_solver, G_loss])

                print(dis_loss, gen_loss)
            except tf.errors.OutOfRangeError:
                break

        some_fake_images = sess.run(fabri_batch, feed_dict={is_training: False})
        fig = plot(some_fake_images[:15, ...])
        plt.show()

        # saver.save(sess, "save_here/save.ckpt")

# with tf.Session() as sess:
#     saver.restore(sess, "save_here/save.ckpt")
#
#     for _ in range(10):
#         # images, labels = iterate(paired_images, train_x, iterator, sess)
#         # images, labels = iterate(paired_images, test_x, iterator, sess)
#
#         sess.run(iterator.initializer, feed_dict={real_images: train_x})
#         while True:
#             try:
#                 _, dis_loss = sess.run([D_solver, D_loss])
#                 _, gen_loss = sess.run([G_solver, G_loss])
#                 # _, _, gen_loss, dis_loss = sess.run(
#                 #     [G_solver, D_solver, G_loss, D_loss]
#                 # )
#                 # print(gen_loss, dis_loss)
#             except tf.errors.OutOfRangeError:
#                 break
#
#         some_fake_images = sess.run(fabri_batch, feed_dict={is_training: False})
#         fig = plot(some_fake_images[:15, ...])
#         plt.show()
#
#         saver.save(sess, "save_here/save.ckpt")
