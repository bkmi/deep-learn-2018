import random

import utils
import network

import tensorflow as tf


timeseries, validate_timeseries, validate_labels = utils.load()

lag = 1
x, y = utils.lag_data(timeseries, lag=lag)
x, y = utils.whiten(x), utils.whiten(y)
val_x, val_y = utils.lag_data(validate_timeseries, lag=lag)
val_x, val_y = utils.whiten(val_x), utils.whiten(val_y)

length_minib = val_x.shape[0]
minibatches = []
for i in range(x.shape[0] - length_minib + 1):
    minibatches.append((x[i:i + length_minib], y[i:i + length_minib]))

timeseries_x = tf.placeholder(tf.float32, shape=[None, length_minib, timeseries.shape[-1]])
timeseries_y = tf.placeholder(tf.float32, shape=[None, length_minib, timeseries.shape[-1]])
loss, encoded, decoded = network.time_lagged_autoencoder(timeseries_x, timeseries_y)
train = tf.train.AdamOptimizer().minimize(loss)

epochs = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        if True:
            random.shuffle(minibatches)
        for j in range(len(minibatches)):
            _, training_loss, training_dim_reduction = sess.run([train, loss, decoded],
                                                                feed_dict={timeseries_x: minibatches[j][0],
                                                                           timeseries_y: minibatches[j][1]})

        validation_loss, validation_dim_reduction = sess.run([loss, decoded],
                                                             feed_dict={timeseries_x: val_x,
                                                                        timeseries_y: val_y})
        print('Validation loss: {}'.format(validation_loss))
        print('Adjusted Rand Index: {}'.format(utils.cluster_compare(val_y. validation_dim_reduction)))
