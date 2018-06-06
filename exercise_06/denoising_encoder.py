import tensorflow as tf
import numpy as np


def gen_data():
    x = np.linspace(-1, 1, 101)
    y = 2 * x + np.random.randn(*x.shape) * 0.2
    return x, y


def model():
    x = tf.placeholder(tf.float32, shape=(None,), name='x')
    y = tf.placeholder(tf.float32, shape=(None,), name='y')

    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(np.random.randn(), name='W')
        y_pred = tf.multiply(w, x)

        loss = tf.reduce_mean(tf.square(y_pred - y))
    return x, y, y_pred, loss

x_data, y_data = gen_data()
x, y, y_pred, loss = model()
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    feed_dict = {x: x_data, y: y_data}
    for _ in range(30):
        loss_value, _ = sess.run([loss, optimizer], feed_dict)

        print(loss_value.mean())

