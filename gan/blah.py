import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as layers
import matplotlib.pylab as plt

from sklearn.preprocessing import LabelBinarizer

framework = tf.contrib.framework

def load_mnist_data():
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    test_x, test_y = test

    train_x, test_x = map(lambda x: x[..., None], [train_x, test_x])
    train_y, test_y = map(lambda x: LabelBinarizer().fit_transform(x), [train_y, test_y])

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = load_mnist_data()


def generator_fn(noise, weight_decay=2.5e-5, is_training=True):
    """Simple generator to produce MNIST images.

    Args:
        noise: A single Tensor representing noise.
        weight_decay: The value of the l2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population
            statistics.

    Returns:
        A generated image in the range [-1, 1].
    """
    with framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)), \
         framework.arg_scope([layers.batch_norm], is_training=is_training,
                             zero_debias_moving_mean=True):
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 256)
        net = tf.reshape(net, [-1, 7, 7, 256])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

        return net


def discriminator_fn(img, unused_conditioning, weight_decay=2.5e-5,
                     is_training=True):
    """Discriminator network on MNIST digits.

    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population
            statistics.

    Returns:
        Logits for the probability that the image is real.
    """
    with framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=tf.nn.leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)

# def visualize_digits(tensor_to_visualize):
#     """Visualize an image once. Used to visualize generator before training.
#
#     Args:
#         tensor_to_visualize: An image tensor to visualize. A python Tensor.
#     """
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         with queues.QueueRunners(sess):
#             images_np = sess.run(tensor_to_visualize)
#     plt.axis('off')
#     plt.imshow(np.squeeze(images_np), cmap='gray')

tfgan = tf.contrib.gan
batch_size = 10
real_images = train_x

noise_dims = 100
# gan_model = tfgan.gan_model(
#     generator_fn,
#     discriminator_fn,
#     real_data=real_images,
#     generator_inputs=tf.random_normal([batch_size, noise_dims]))

# Sanity check that generated images before training are garbage.
# check_generated_digits = tfgan.eval.image_reshaper(
#     gan_model.generated_data[:20,...], num_cols=10)
# visualize_digits(check_generated_digits)

innoise = tf.random_normal([batch_size, noise_dims])
a = generator_fn(innoise)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))