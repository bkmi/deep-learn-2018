import random

import utils
import network

import tensorflow as tf


timeseries, validate_timeseries, validate_labels = utils.load()

lag = 1
assert lag > 0
x, y = utils.lag_data(timeseries, lag=lag)
x, y = utils.whiten(x), utils.whiten(y)
val_x, val_y = utils.lag_data(validate_timeseries, lag=0)
val_x, val_y = utils.whiten(val_x), utils.whiten(val_y)

# length_minib = val_x.shape[0]
# minibatches = []
# for i in range(x.shape[0] - length_minib + 1):
#     minibatches.append((x[i:i + length_minib], y[i:i + length_minib]))

timeseries_x = tf.placeholder(tf.float32, shape=[None, timeseries.shape[-1]])
timeseries_y = tf.placeholder(tf.float32, shape=[None, timeseries.shape[-1]])
loss, encoded, decoded = network.time_lagged_autoencoder(timeseries_x, timeseries_y)
train = tf.train.AdamOptimizer().minimize(loss)

epochs = 1500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        _, _ = sess.run([train, loss],
                        feed_dict={timeseries_x: x,
                                   timeseries_y: y})

        if i % 500 == 0:
            validation_loss, validation_dim_reduction = sess.run([loss, encoded],
                                                                 feed_dict={timeseries_x: val_x,
                                                                            timeseries_y: val_y})
            print('Validation loss: {}'.format(validation_loss))
            score = utils.cluster_compare(validate_labels, validation_dim_reduction)
            print('Adjusted Rand Index: {}'.format(score))

    encoded_timeseries = sess.run(encoded,
                                  feed_dict={timeseries_x: timeseries,
                                             timeseries_y: timeseries})

    pred_timeseries_y = utils.cluster(encoded_timeseries)
    print(pred_timeseries_y.labels_)
    utils.save(pred_timeseries_y.labels_)
