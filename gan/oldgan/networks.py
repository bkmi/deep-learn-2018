import tensorflow as tf

layers = tf.contrib.layers

def generator(noise, weight_decay=2.5e-5):
  """Core MNIST generator.

  Args:
    noise: A 2D Tensor of shape [batch size, noise dim].
    weight_decay: The value of the l2 weight decay.

  Returns:
    A generated image in the range [-1, 1].
  """
  with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.fully_connected(noise, 1024)
    net = layers.fully_connected(net, 7 * 7 * 128)
    net = tf.reshape(net, [-1, 7, 7, 128])
    net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
    net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
    # Make sure that generator output is in the same range as `inputs`
    # ie [-1, 1].
    net = layers.conv2d(
        net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

    return net


_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)


def discriminator(img, weight_decay):
  """Core MNIST discriminator.

  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    weight_decay: The L2 weight decay.

  Returns:
    Final fully connected discriminator layer. [batch_size, 1024].
  """
  with tf.contrib.framework.arg_scope(
      [layers.conv2d, layers.fully_connected],
      activation_fn=_leaky_relu, normalizer_fn=None,
      weights_regularizer=layers.l2_regularizer(weight_decay),
      biases_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.conv2d(img, 64, [4, 4], stride=2)
    net = layers.conv2d(net, 128, [4, 4], stride=2)
    net = layers.flatten(net)
    net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

    return net