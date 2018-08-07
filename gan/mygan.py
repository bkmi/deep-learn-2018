import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.preprocessing import LabelBinarizer


def load_mnist_data():
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    test_x, test_y = test

    train_x, test_x = map(lambda x: x[..., None], [train_x, test_x])
    train_y, test_y = map(lambda x: LabelBinarizer().fit_transform(x), [train_y, test_y])

    return train_x, train_y, test_x, test_y


def generator(noise, reuse=False):
    with tf.variable_scope(name_or_scope="generator", reuse=reuse):
        x = tf.layers.dense(
            noise,
            128,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            name='relu',
            reuse=reuse
        )
        x = tf.layers.dense(
            x,
            28 * 28,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            name='sig',
            reuse=reuse
        )
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
    return x


def discriminator(img, reuse=False):
    with tf.variable_scope(name_or_scope="discriminator", reuse=reuse):
        x = tf.layers.flatten(img)
        x = tf.layers.dense(
            x,
            128,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            name='relu',
            reuse=reuse
        )
        x = tf.layers.dense(
            x,
            1,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            name='sig',
            reuse=reuse
        )
    return x


def create_dataset(real_images):
    with tf.name_scope('dataset_iteratior'):
        dataset = tf.data.Dataset.from_tensor_slices(real_images)
        dataset = dataset.shuffle(buffer_size=train_x.shape[0])
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=batch_size))

        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

    return next_element, iterator


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
batch_size, noise_dims = 128, 100
epochs = 500

# real_images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='real_images')
# real_batch, iterator = create_dataset(real_images)
real_batch = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])

# noise = tf.random_uniform(shape=[batch_size, noise_dims], minval=-1, maxval=1, name='noise')
noise = tf.placeholder(shape=[None, noise_dims], dtype=tf.float32)
fabri_batch = generator(noise)

D_real = discriminator(real_batch, reuse=False)
D_fake = discriminator(fabri_batch, reuse=True)

tf.summary.image('real', real_batch, 4)
tf.summary.image('fabr', fabri_batch, 4)

D_loss = -tf.reduce_mean(tf.log(D_real + 1e-12) + tf.log(1. - D_fake + 1e-12), name='discriminator_loss')
G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-12), name='generator_loss')

# Now I need to keep track of which variables to update.
# start with this easy example then later implement the better loss function
# https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
# better loss: https://www.alexirpan.com/2017/02/22/wasserstein-gan.html
D_solver = tf.train.AdamOptimizer().minimize(
    D_loss,
    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
    name='discriminator_solver'
)
G_solver = tf.train.AdamOptimizer().minimize(
    G_loss,
    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'),
    name='generator_solver'
)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    tf.summary.histogram(i.name, i)

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

saver = tf.train.Saver()
merged_summary = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('/tmp/mygan/1')
    writer.add_graph(sess.graph)

    for i in range(epochs):
        # sess.run(iterator.initializer, feed_dict={real_images: train_x})

        for j in range(train_x.shape[0] // 128):
            X_mb, _ = mnist.train.next_batch(batch_size)
            X_mb = np.reshape(X_mb, newshape=[-1, 28, 28, 1])

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={real_batch: X_mb,
                                                                     noise: sample_Z(batch_size, noise_dims)})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={noise: sample_Z(batch_size, noise_dims)})

        s = sess.run(merged_summary, feed_dict={real_batch: X_mb,
                                                noise: sample_Z(batch_size, noise_dims)})
        writer.add_summary(s, i)


        # s = sess.run(merged_summary)
        # writer.add_summary(s, i)
        #
        # while True:
        #     try:
        #         _, dis_loss, samps1 = sess.run([D_solver, D_loss, noise])
        #         _, gen_loss, samps2 = sess.run([G_solver, G_loss, noise])
        #
        #         # print(np.isclose(samps1, samps2))
        #         # print(dis_loss, gen_loss)
        #     except tf.errors.OutOfRangeError:
        #         break

        some_fake_images = sess.run(fabri_batch, feed_dict={noise: sample_Z(16, noise_dims)})
        fig = plot(some_fake_images)
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
