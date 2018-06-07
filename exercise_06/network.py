import tensorflow as tf


layers = tf.contrib.layers


def encoder(flattened_image):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.relu):
        net = layers.fully_connected(flattened_image, 128)
        net = layers.fully_connected(net, 64)
        net = layers.fully_connected(net, 32)
    # net = layers.fully_connected(net, 32, activation_fn=None)
    return net


def decoder(representation):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.relu):
        net = layers.fully_connected(representation, 64)
        net = layers.fully_connected(net, 128)
    net = layers.fully_connected(net, 28*28, activation_fn=tf.nn.sigmoid)
    return net


def denoising_autoencoder(image):
    """image.shape = [batch_size, x, y, channels]"""
    flat_image = tf.reshape(image, [-1, 28 * 28])
    encoded = encoder(flat_image)
    decoded = decoder(encoded)
    return encoded, decoded


def autoencoder(image):
    """image.shape = [batch_size, x, y, channels]"""
    flat_image = tf.reshape(image, [-1, 28 * 28])
    encoded = encoder(flat_image)
    decoded = decoder(encoded)
    loss = tf.reduce_mean(tf.squared_difference(flat_image, decoded))
    return loss, encoded, decoded
