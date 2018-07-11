import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as layers
import matplotlib.pylab as plt

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


def generator(noise, weight_decay=2.5e-5):
    with tf.contrib.framework.arg_scope([layers.fully_connected, layers.conv2d_transpose],
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm,
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
    net = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.sigmoid)

    return net


def discriminator(img, weight_decay=4e-5):
    with tf.contrib.framework.arg_scope([layers.conv2d, layers.fully_connected],
                                        activation_fn=tf.nn.leaky_relu,
                                        normalizer_fn=None,
                                        weights_regularizer=layers.l2_regularizer(weight_decay),
                                        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
    net = layers.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)

    return net


def mix_real_and_fabricated(real_images, fabri_images):
    dataset = tf.data.Dataset.from_tensor_slices(real_images)
    dataset = dataset.shuffle(buffer_size=train_x.shape[0])
    dataset = dataset.batch(batch_size=batch_size // 2)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    labeled_images = (tf.concat([tf.cast(next_element, fabri_images.dtype), fabri_images], axis=0),
                      tf.concat([tf.ones(batch_size // 2), tf.zeros(batch_size // 2)], axis=0))
    labeled_images = tuple(map(lambda x: tf.random_shuffle(x, seed=42), labeled_images))
    return labeled_images, iterator


def pair_real_and_fabricated(real_images, fabri_images):
    dataset = tf.data.Dataset.from_tensor_slices(real_images)
    dataset = dataset.shuffle(buffer_size=train_x.shape[0])
    dataset = dataset.batch(batch_size=batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # paired_images = (tf.concat([tf.cast(next_element, fabri_images.dtype), fabri_images], axis=0),
    #                  tf.concat([tf.ones(batch_size), tf.zeros(batch_size)], axis=0))
    paired_images = (next_element, fabri_images)
    paired_images = tuple(map(lambda x: tf.random_shuffle(x, seed=42), paired_images))
    return paired_images, iterator


def iterate(predicate, images, iterator, session):
    session.run(iterator.initializer, feed_dict={real_images: images})
    while True:
        try:
            result = session.run(predicate)
        except tf.errors.OutOfRangeError:
            break
    return result


train_x, train_y, test_x, test_y = load_mnist_data()
batch_size, noise_dims = 32, 64
epochs = 10

real_images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
fabri_images = generator(tf.random_normal([batch_size, noise_dims]))
paired_images, iterator = pair_real_and_fabricated(real_images, fabri_images)

D_real = discriminator(paired_images[0])
D_fake = discriminator(paired_images[1])

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

# Now I need to keep track of which variables to update.
# start with this easy example then later implement the better loss function
# https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
# better loss: https://www.alexirpan.com/2017/02/22/wasserstein-gan.html
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(epochs):
        images, labels = iterate(paired_images, train_x, iterator, sess)
        images, labels = iterate(paired_images, test_x, iterator, sess)
