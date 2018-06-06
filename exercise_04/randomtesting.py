import numpy as np
import tensorflow as tf

# b = tf.Variable(tf.zeros((100,)))
# W = tf.Variable(tf.random_uniform((784, 100), -1, 1))
# x = tf.placeholder(tf.float32, (100, 784))
# h = tf.nn.relu(tf.matmul(x, W) + b)
#
# with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
#     init = tf.global_variables_initializer()
#     sess.run(h, {x: np.random.rand(100, 784)})
#     # print(h.eval())

# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
    # Assemble a graph consisting of the following three operations:
    #   * Two tf.constant operations to create the operands.
    #   * One tf.add operation to add the two operands.
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    my_sum = tf.add(x, y, name="x_y_sum")


    # Now create a session.
    # The session will run the default graph.
    with tf.Session() as sess:
        print(my_sum.eval())
