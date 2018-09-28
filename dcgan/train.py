import tensorflow as tf

from data import create_mnist, create_dataset, SaverHelper
from network import MNISTDCGAN
from pathlib import Path


(x_train, y_train), (x_test, y_test) = create_mnist()
batch_size = 128
epochs = 1000

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

gan = MNISTDCGAN(image, latent, learning_rate=0.0002, beta1=0.5, log_bias=1e-12)

saver = tf.train.Saver()
saver_helper = SaverHelper()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    valid_iterator = valid_dataset.make_one_shot_iterator()
    valid_handle = sess.run(valid_iterator.string_handle())

    counter = 0
    for ep in range(epochs):
        counter += 1

        train_iterator = train_dataset.make_one_shot_iterator()
        train_handle = sess.run(train_iterator.string_handle())

        if True and (ep % 100 == 0):
            saver.save(sess, str(Path(saver_helper.models_dir, f'save{ep}.ckpt')))
            gan.write_grid(sess, path=Path(saver_helper.images_dir, f'epoch_{ep}.png'))

        while True:
            try:
                d_loss, g_loss = gan.train_batch(sess, feed_dict={handle: train_handle})
            except tf.errors.OutOfRangeError:
                break

        if True and (ep % 100 == 0):
            saver.save(sess, str(Path(saver_helper.models_dir, f'save{ep}.ckpt')))
            gan.write_grid(sess, path=Path(saver_helper.images_dir, f'epoch_{ep}.png'))

        print(f'd_loss: {d_loss}, g_loss: {g_loss}')
