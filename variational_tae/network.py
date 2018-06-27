import tensorflow as tf


layers = tf.contrib.layers


def encoder(timeseries, dim_latent_space, training_status=False):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.leaky_relu):
        net = layers.fully_connected(timeseries, 200)
        net = tf.layers.dropout(net)
        net = layers.fully_connected(net, 100)
        net = tf.layers.dropout(net)
    # encoded_timeseries = tf.cond(training_status,
    #                            layers.fully_connected(net, 1, activation_fn=tf.nn.leaky_relu),
    #                            layers.fully_connected(net, 1, activation_fn=None))
    encoded_mean = layers.fully_connected(net, dim_latent_space, activation_fn=None)
    encoded_stdd = layers.fully_connected(net, dim_latent_space, activation_fn=tf.nn.relu)
    encoded_stdd = tf.add(encoded_stdd, 1e-6)
    return encoded_mean, encoded_stdd


def decoder(encoded, output_size):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.relu):
        net = layers.fully_connected(encoded, 100)
        net = tf.layers.dropout(net)
        net = layers.fully_connected(net, 200)
        net = tf.layers.dropout(net)
    decoded_timeseries = layers.fully_connected(net, output_size, activation_fn=None)
    return decoded_timeseries


def sample_encoded_space(encoded_mean, encoded_stdd, sample_shape):
    samples = tf.random_normal(shape=sample_shape)
    encoded = encoded_mean + encoded_stdd * samples
    return encoded


def time_lagged_variational_autoencoder(timeseries_x, timeseries_y, count_timesteps, dim_latent_space=1, **kwargs):
    encoded_mean, encoded_stdd = encoder(timeseries_x, dim_latent_space=dim_latent_space, **kwargs)

    sample_shape = tf.concat([count_timesteps, [dim_latent_space]], axis=0)
    encoded = sample_encoded_space(encoded_mean, encoded_stdd, sample_shape)

    decoded = decoder(encoded, int(timeseries_x.shape[-1]))
    reconstruction_loss = tf.reduce_mean(tf.squared_difference(timeseries_y, decoded))
    kl_divergence = -0.5 * tf.reduce_sum(
        tf.square(encoded_mean) + tf.square(encoded_stdd) - tf.log(tf.square(encoded_stdd)) - 1)
    loss = -kl_divergence + reconstruction_loss
    return loss, encoded_mean, encoded_stdd, encoded, decoded
