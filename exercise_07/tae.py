import tensorflow as tf
import numpy as np

import utils
import network


data, validate_x, validate_y = utils.load()
data_whitened = utils.whiten(data)
validate_x_whitened = utils.whiten(validate_x)
training_data, test_data = np.split(data_whitened, 2)

timeseries = tf.placeholder(tf.float32, shape=[None, training_data.shape[-1]])
training_status = tf.placeholder(tf.bool, shape=[1])
loss, encoded, decoded = network.time_lagged_autoencoder(timeseries, training_status)
training_step = tf.train.AdamOptimizer().minimize(loss)
