import tensorflow as tf

from data import create_mnist, create_dataset, SaverHelper
from network import MNISTDCGAN
from pathlib import Path
from absl import flags, app


GANTYPE_HELP = 'dcgan, wgan, iwgan'
flags.DEFINE_string('gantype', 'dcgan', GANTYPE_HELP)
flags.DEFINE_bool('savepics', False, 'Save pictures?')
flags.DEFINE_bool('savemodels', False, 'Save models?')
flags.DEFINE_integer('epochs', 1000, 'Integer number of training epochs.')
flags.DEFINE_bool('gpu', True, 'Use gpu?')

FLAGS = flags.FLAGS


def main(_):
    (x_train, y_train), (x_test, y_test) = create_mnist()
    batch_size = 64

    train_dataset, valid_dataset = [
        create_dataset(
            x,
            y,
            batch_size=z,
            buffer_size=1000,
            repeat=w
        ) for x, y, z, w in zip((x_train, x_test), (y_train, y_test), (batch_size, 25), (False, True))
    ]
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    image, label = iterator.get_next()

    latent = tf.random_uniform(shape=[batch_size, 100], minval=-1, maxval=1)

    if FLAGS.gantype == 'dcgan':
        gan = MNISTDCGAN(image, latent, learning_rate=0.0005, beta1=0.5, log_bias=1e-12)
        print('DCGAN')
    elif FLAGS.gantype == 'wgan':
        gan = MNISTDCGAN(image, latent, learning_rate=5e-5)
        print('WGAN')
    elif FLAGS.gantype == 'iwgan':
        gan = MNISTDCGAN(image, latent, learning_rate=1e-4, beta1=0.5, beta2=0.9)
        print('IWGAN')
    else:
        raise ValueError('Must choose between :' + GANTYPE_HELP)

    saver = tf.train.Saver()
    if FLAGS.savepics or FLAGS.savemodels:
        saver_helper = SaverHelper(mkdir=True)
        print('making save dir')
    else:
        saver_helper = SaverHelper(mkdir=False)
        print('NOT making save dir')

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    if FLAGS.gpu:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        valid_iterator = valid_dataset.make_one_shot_iterator()
        valid_handle = sess.run(valid_iterator.string_handle())

        for ep in range(FLAGS.epochs):
            train_iterator = train_dataset.make_one_shot_iterator()
            train_handle = sess.run(train_iterator.string_handle())

            while True:
                try:
                    d_loss, g_loss = gan.train_batch(sess, feed_dict={handle: train_handle})
                except tf.errors.OutOfRangeError:
                    break

            if FLAGS.savemodels and (ep % 100 == 0):
                saver.save(sess, str(Path(saver_helper.models_dir, f'save{ep}.ckpt')))
            if FLAGS.savepics and (ep % 100 == 0):
                gan.write_grid(sess, path=Path(saver_helper.images_dir, f'epoch_{ep}.png'))

            print(f'd_loss: {d_loss}, g_loss: {g_loss}')


if __name__ == '__main__':
    app.run(main)
