import tensorflow as tf


layers = tf.contrib.layers


def encoder(flattened_image):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.tanh):
        net = layers.fully_connected(flattened_image, 50)
        net = layers.fully_connected(net, 50)
    net = layers.fully_connected(net, 2, activation_fn=None)
    return net


def decoder(representation):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.tanh):
        net = layers.fully_connected(representation, 50)
        net = layers.fully_connected(net, 50)
    net = layers.fully_connected(net, 28*28, activation_fn=None)
    return net


def autoencoder(image):
    """image.shape = [batch_size, x, y, channels]"""
    flat_image = tf.reshape(image, [-1, 28 * 28])
    encoded = encoder(flat_image)
    decoded = decoder(encoded)
    loss = tf.reduce_mean(tf.squared_difference(flat_image, decoded))
    return loss, encoded, decoded


def denoising_autoencoder(image, noise_cov):
    """image.shape = [batch_size, x, y, channels]"""
    distorted_image = image + tf.random_normal(tf.shape(image), mean=0.0, stddev=noise_cov)
    flat_image = tf.reshape(distorted_image, [-1, 28 * 28])
    encoded = encoder(flat_image)
    decoded = decoder(encoded)
    loss = tf.reduce_mean(tf.squared_difference(tf.reshape(image, [-1, 28 * 28]),
                                                decoded))
    return loss, encoded, decoded
