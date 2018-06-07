import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import inout
import network


def remaining_noise(clean_data, noisy_data):
    # return np.sqrt(np.mean((noisy_data - clean_data) ** 2))
    return np.linalg.norm(clean_data - noisy_data)


# Data
training_images_clean, validation_images_noisy, validation_images_clean, test_images_noisy = inout.load()

# Calculate Standard Deviation of Error
sigma = remaining_noise(validation_images_clean, validation_images_noisy)

# Model
clean_image = tf.placeholder(tf.float32, shape=[None, 1, 28, 28])
noisy_image = tf.placeholder(tf.float32, shape=[None, 1, 28, 28])
encoded, decoded = network.denoising_autoencoder(noisy_image)
loss = tf.keras.backend.binary_crossentropy(tf.reshape(clean_image, [-1, 28 * 28]), decoded)
# loss = tf.reduce_mean((tf.reshape(clean_image, [-1, 28 * 28]) - decoded) ** 2)
train_step = tf.train.AdadeltaOptimizer().minimize(loss)

# Train
count_epochs = 50
images_per_batch = 40
count_batches = int(training_images_clean.shape[0] / images_per_batch)
training_batches_clean = np.split(training_images_clean, count_batches)
training_batches_noisy = np.split(training_images_clean + sigma * np.random.randn(*training_images_clean.shape),
                                  count_batches)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    validation_noise_record = 1
    validation_feed = {noisy_image: validation_images_noisy}
    test_feed = {noisy_image: test_images_noisy}
    for i in range(count_epochs * count_batches):
        ind = i % count_batches

        training_feed = {noisy_image: training_batches_noisy[ind], clean_image: training_batches_clean[ind]}
        if i % 500 == 0:
            train_loss = sess.run(loss, feed_dict=training_feed)
            print("Step: %d. Loss: %g" % (i, train_loss.mean()))

        if i % 2000 == 0:
            validation_decoded = sess.run(decoded, feed_dict=validation_feed)
            validation_noise_rem = remaining_noise(np.reshape(validation_images_clean, [-1, 28 * 28]),
                                                   validation_decoded)
            print("Remaining Validation Noise: ", validation_noise_rem)

            if validation_noise_rem < validation_noise_record:
                validation_noise_record = validation_noise_rem

                if validation_noise_record < 0.2:
                    test_decoded = sess.run(decoded, feed_dict=test_feed)
                    test_decoded_image = np.reshape(test_decoded, [2000, 1, 28, 28])
                    inout.save(test_decoded_image)

                    break

        if i % 20000 == 0:
            plt.imshow(np.reshape(validation_decoded, validation_images_clean.shape)[0, 0])
            plt.show()
            plt.imshow(validation_images_clean[0, 0])
            plt.show()

        train_step.run(feed_dict=training_feed)
