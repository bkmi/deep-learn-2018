import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from networks import generator, discriminator
from datasets import dataset_factory as datasets
from datasets import download_and_convert_mnist


# Set Flags
flags = tf.flags

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/mnist/',
                    'Directory where to write event logs.')

dataset_dir = Path('mnist_data')
dataset_dir.mkdir(exist_ok=True)
flags.DEFINE_string('dataset_dir', str(dataset_dir.resolve()), 'Location of data.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_string(
    'gan_type', 'unconditional',
    'Either `unconditional`, `conditional`, or `infogan`.')

flags.DEFINE_integer(
    'grid_size', 5, 'Grid size for image visualization.')


flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector.')

FLAGS = flags.FLAGS

# Restrict GPU memory
config = tf.ConfigProto()  # device_count = {'GPU': 0}
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True


def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.

    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.contrib.slim.queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    plt.imshow(np.squeeze(images_np), cmap='gray')

# Data
def provide_data(split_name, batch_size, dataset_dir, num_readers=1,
                 num_threads=1):
    """Provides batches of MNIST digits.

    Args:
      split_name: Either 'train' or 'test'.
      batch_size: The number of images in each batch.
      dataset_dir: The directory where the MNIST data can be found.
      num_readers: Number of dataset readers.
      num_threads: Number of prefetching threads.

    Returns:
      images: A `Tensor` of size [batch_size, 28, 28, 1]
      one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
        each row has a single element set to one and the rest set to zeros.
      num_samples: The number of total samples in the dataset.

    Raises:
      ValueError: If `split_name` is not either 'train' or 'test'.
    """
    dataset = datasets.get_dataset('mnist', split_name, dataset_dir=dataset_dir)
    provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size,
        shuffle=(split_name == 'train'))
    [image, label] = provider.get(['image', 'label'])

    # Preprocess the images.
    image = (tf.to_float(image) - 128.0) / 128.0

    # Creates a QueueRunner for the pre-fetching operation.
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size)

    one_hot_labels = tf.one_hot(labels, dataset.num_classes)
    return images, one_hot_labels, dataset.num_samples


def train():
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            images, one_hot_labels, _ = provide_data(
                'train', FLAGS.batch_size, FLAGS.dataset_dir, num_threads=4)

    # Model
    noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])
    gan_model = tf.contrib.gan.gan_model(
        generator_fn=generator,
        discriminator_fn=discriminator,
        real_data=images,
        generator_inputs=noise
    )
    tf.contrib.gan.eval.add_gan_model_image_summaries(gan_model, FLAGS.grid_size)

    # Cost function & Optimization
    # Get the GANLoss tuple. You can pass a custom function, use one of the
    # already-implemented losses from the losses library, or use the defaults.
    with tf.name_scope('loss'):
        gan_loss = tf.contrib.gan.gan_loss(
            gan_model,
            gradient_penalty_weight=1.0,
            add_summaries=True)
        tf.contrib.gan.eval.add_regularization_loss_summaries(gan_model)

    # Get the GANTrain ops using custom optimizers.
    with tf.name_scope('train'):
        gen_lr, dis_lr = 1e-3, 1e-4
        train_ops = tf.contrib.gan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
            discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5),
            summarize_gradients=True,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # Training
    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')
    if FLAGS.max_number_of_steps == 0:
        return tf.contrib.gan.gan_train(
            train_ops,
            hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
                   tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
            logdir=FLAGS.train_log_dir,
            get_hooks_fn=tf.contrib.gan.get_joint_train_hooks())

if __name__ == '__main__':
    if not tf.gfile.Exists(FLAGS.dataset_dir):
        tf.gfile.MakeDirs(FLAGS.dataset_dir)

    download_and_convert_mnist.run(FLAGS.dataset_dir)

    tf.reset_default_graph()

    # Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
    # for forward and backwards propogation.
    batch_size = 32
    with tf.device('/cpu:0'):
        images, _, _ = provide_data('train', batch_size, FLAGS.dataset_dir)

    # Sanity check that we're getting images.
    imgs_to_visualize = tf.contrib.gan.eval.image_reshaper(images[:20, ...], num_cols=10)
    visualize_digits(imgs_to_visualize)

    noise_dims = 64
    gan_model = tfgan.gan_model(
        generator_fn,
        discriminator_fn,
        real_data=images,
        generator_inputs=tf.random_normal([batch_size, noise_dims]))

    # Sanity check that generated images before training are garbage.
    generated_data_to_visualize = tfgan.eval.image_reshaper(
        gan_model.generated_data[:20, ...], num_cols=10)
    visualize_digits(generated_data_to_visualize)

    # We have the option to train the discriminator more than one step for every
    # step of the generator. In order to do this, we use a `GANTrainSteps` with
    # desired values. For this example, we use the default 1 generator train step
    # for every discriminator train step.
    train_step_fn = tf.contrib.gan.get_sequential_train_steps()

    global_step = tf.train.get_or_create_global_step()
    loss_values, mnist_score_values  = [], []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.contrib.slim.queues.QueueRunners(sess):
            start_time = time.time()
            for i in range(801):
                cur_loss, _ = train_step_fn(
                    sess, gan_train_ops, global_step, train_step_kwargs={})
                loss_values.append((i, cur_loss))
                if i % 100 == 0:
                    mnist_score_values.append((i, sess.run(eval_score)))
                if i % 200 == 0:
                    print('Current loss: %f' % cur_loss)
                    print('Current MNIST score: %f' % mnist_score_values[-1][1])
                    visualize_training_generator(
                        i, start_time, sess.run(generated_data_to_visualize))
