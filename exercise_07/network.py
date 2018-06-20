import tensorflow as tf


layers = tf.contrib.layers


def encoder(timeseries, training_status=False):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.leaky_relu):
        net = layers.fully_connected(timeseries, 200)
        net = tf.layers.dropout(net)
        net = layers.fully_connected(net, 100)
        net = tf.layers.dropout(net)
    # encoded_timeseries = tf.cond(training_status,
    #                            layers.fully_connected(net, 1, activation_fn=tf.nn.leaky_relu),
    #                            layers.fully_connected(net, 1, activation_fn=None))
    encoded_timeseries = layers.fully_connected(net, 1, activation_fn=None)
    return encoded_timeseries


def decoder(encoded_timeseries, output_size):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.relu):
        net = layers.fully_connected(encoded_timeseries, 100)
        net = tf.layers.dropout(net)
        net = layers.fully_connected(net, 200)
        net = tf.layers.dropout(net)
    decoded_timeseries = layers.fully_connected(net, output_size, activation_fn=None)
    return decoded_timeseries


def time_lagged_autoencoder(timeseries_x, timeseries_y, **kwargs):
    encoded = encoder(timeseries_x, **kwargs)
    decoded = decoder(encoded, int(timeseries_x.shape[-1]))
    loss = tf.reduce_mean(tf.squared_difference(timeseries_y, decoded))
    return loss, encoded, decoded
