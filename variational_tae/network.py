import tensorflow as tf


layers = tf.contrib.layers


def encoder(timeseries, dim_latent_space, training_status):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.elu):
        net = layers.fully_connected(timeseries, 200)
        net = tf.layers.dropout(net, training=training_status)
        net = layers.fully_connected(net, 100)
        net = tf.layers.dropout(net, training=training_status)
    # encoded_timeseries = tf.cond(training_status,
    #                            layers.fully_connected(net, 1, activation_fn=tf.nn.leaky_relu),
    #                            layers.fully_connected(net, 1, activation_fn=None))
    encoded_mean = layers.fully_connected(net, dim_latent_space, activation_fn=None)
    encoded_log_stdd = layers.fully_connected(net, dim_latent_space, activation_fn=None)
    encoded_stdd = tf.sqrt(tf.exp(encoded_log_stdd))  # sqrt(exp(log_sigma) ** 2) * N(0,1) = sigma * N(0,1)
    return encoded_mean, encoded_stdd


def decoder(encoded, output_size, training_status):
    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.elu):
        net = layers.fully_connected(encoded, 100)
        net = tf.layers.dropout(net, training=training_status)
        net = layers.fully_connected(net, 200)
        net = tf.layers.dropout(net, training=training_status)
    decoded_timeseries = layers.fully_connected(net, output_size, activation_fn=None)
    return decoded_timeseries


def sample_encoded_space(encoded_mean, encoded_stdd, sample_shape):
    samples = tf.random_normal(shape=sample_shape)
    encoded = encoded_mean + encoded_stdd * samples
    return encoded


def time_lagged_variational_autoencoder(timeseries_x,
                                        timeseries_y,
                                        count_timesteps,
                                        training_status):
    dim_latent_space = 1
    encoded_mean, encoded_stdd = encoder(timeseries_x, dim_latent_space, training_status)

    sample_shape = tf.concat([count_timesteps, [dim_latent_space]], axis=0)
    encoded = sample_encoded_space(encoded_mean, encoded_stdd, sample_shape)

    decoded = decoder(encoded, int(timeseries_x.shape[-1]), training_status)

    # Real loss Method
    # reconstruction error is over multiple samples
    # dataset error is N/M sum(term,
    # reconstruction_loss = tf.reduce_sum(tf.reduce_mean(tf.squared_difference(timeseries_y, decoded), 1))
    reconstruction_loss = tf.reduce_sum(tf.squared_difference(timeseries_y, decoded), axis=1)
    kl_divergence = tf.reduce_sum(
        0.5 * (tf.square(encoded_mean) + tf.square(encoded_stdd) - tf.log(tf.square(encoded_stdd)) - 1), axis=1)
    loss = kl_divergence + reconstruction_loss
    loss = tf.Print(loss, [tf.reduce_sum(kl_divergence), tf.reduce_sum(reconstruction_loss)])
    return loss, encoded_mean, encoded_stdd, encoded, decoded
