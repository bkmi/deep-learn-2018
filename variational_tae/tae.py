import random

import variational_tae.utils as utils
import variational_tae.network as network

import tensorflow as tf


timeseries, validate_timeseries, validate_labels = utils.load()

lag = 1
assert lag > 0
x, y = utils.lag_data(timeseries, lag=lag)
x, y = utils.whiten(x), utils.whiten(y)
val_x, val_y = utils.lag_data(validate_timeseries, lag=0)  # maybe with lag
val_x, val_y = utils.whiten(val_x), utils.whiten(val_y)

# length_minib = val_x.shape[0]
# minibatches = []
# for i in range(x.shape[0] - length_minib + 1):
#     minibatches.append((x[i:i + length_minib], y[i:i + length_minib]))

timeseries_x = tf.placeholder(tf.float32, shape=[None, timeseries.shape[-1]])
timeseries_y = tf.placeholder(tf.float32, shape=[None, timeseries.shape[-1]])
count_timesteps = tf.placeholder(tf.int32, shape=[1])
loss, encoded, decoded = network.time_lagged_variational_autoencoder(timeseries_x, timeseries_y, count_timesteps)
train = tf.train.AdamOptimizer().minimize(loss)

epochs = 1500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        _, _, enc, dec = sess.run([train, loss, encoded, decoded],
                                  feed_dict={timeseries_x: x,
                                             timeseries_y: y,
                                             count_timesteps: [x.shape[0]]})

        if i % 500 == 0:
            validation_loss, validation_dim_reduction = sess.run([loss, encoded],  # try before sampling
                                                                 feed_dict={timeseries_x: val_x,
                                                                            timeseries_y: val_y,
                                                                            count_timesteps: [val_x.shape[0]]})
            print('Validation loss: {}'.format(validation_loss))
            score = utils.cluster_compare(validate_labels, validation_dim_reduction)
            print('Adjusted Rand Index: {}'.format(score))

    encoded_timeseries = sess.run(encoded,
                                  feed_dict={timeseries_x: timeseries,
                                             timeseries_y: timeseries,
                                             count_timesteps: [timeseries.shape[0]]})

    pred_timeseries_y = utils.cluster(encoded_timeseries)
    print(pred_timeseries_y.labels_)
    utils.save(pred_timeseries_y.labels_)
