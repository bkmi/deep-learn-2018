import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as layers
import matplotlib.pylab as plt

from sklearn.preprocessing import LabelBinarizer

from tensorflow.contrib.gan import gan_model

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
        net = tf.Print(net, [net, tf.reduce_mean(net)])
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.Print(net, [net, tf.reduce_mean(net)])
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = tf.Print(net, [net, tf.reduce_mean(net)])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = tf.Print(net, [net, tf.reduce_mean(net)])
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
    net = layers.fully_connected(net, 2, activation_fn=tf.nn.softmax)

    return net

train_x, train_y, test_x, test_y = load_mnist_data()
batch_size, noise_dims = 32, 64

gen = generator(tf.random_normal([batch_size, noise_dims]))

real_images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
dis = discriminator(real_images)

# learn = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

