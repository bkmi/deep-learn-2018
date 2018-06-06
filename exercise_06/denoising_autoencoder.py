import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import inout
import network


def remaining_noise(clean_data, noisy_data):
    return np.sqrt(np.mean((noisy_data - clean_data) ** 2))


# Data
training_images_clean, validation_images_noisy, validation_images_clean, test_images_noisy = inout.load()

# Calculate Standard Deviation of Error
sigma = remaining_noise(validation_images_clean, validation_images_noisy)

# Model
input_tensor = tf.placeholder(tf.float32, shape=[None, 1, 28, 28])
loss, encoded, decoded = network.denoising_autoencoder(input_tensor, noise_cov=sigma)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Train
count_epochs = 500
images_per_batch = 20
count_batches = int(training_images_clean.shape[0] / images_per_batch)
training_batches = np.split(training_images_clean, count_batches)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    validation_noise_record = 1
    validation_feed = {input_tensor: validation_images_noisy}
    test_feed = {input_tensor: test_images_noisy}
    for i in range(count_epochs * count_batches):
        ind = i % count_batches

        training_feed = {input_tensor: training_batches[ind]}
        if i % 500 == 0:
            train_loss = sess.run(loss, feed_dict=training_feed)
            print("Step: %d. Loss: %g" % (i, train_loss))

        if i % 2000 == 0:
            validation_decoded = sess.run(decoded, feed_dict=validation_feed)
            validation_noise_rem = remaining_noise(np.reshape(validation_images_clean, [-1, 28 * 28]),
                                                   validation_decoded)
            print("Remaining Validation Noise: ", validation_noise_rem)

            if validation_noise_rem < validation_noise_record:
                validation_noise_record = validation_noise_rem

                if validation_noise_record < 0.21:
                    test_decoded = sess.run(decoded, feed_dict=test_feed)
                    test_decoded_image = np.reshape(test_decoded, [2000, 1, 28, 28])
                    plt.imshow(test_decoded_image[0, 0])
                    plt.show()
                    plt.imshow(test_images_noisy[0, 0])
                    plt.show()
                    inout.save(test_decoded_image)

                    break

        train_step.run(feed_dict=training_feed)
