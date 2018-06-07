import inout
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from tensorflow import ConfigProto, Session


# (x_train, _), (x_test, _) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
#
# noise_factor = 0.5
# x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
# x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
#
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# Data
def remaining_noise(clean_data, noisy_data):
    return np.sqrt(np.mean((noisy_data - clean_data) ** 2))


training_images_clean, validation_images_noisy, validation_images_clean, test_images_noisy = inout.load()
sigma = remaining_noise(validation_images_clean, validation_images_noisy)
training_images_noisy = training_images_clean + sigma * np.random.randn(*training_images_clean.shape)

x_train = np.moveaxis(training_images_clean, 1, -1)
x_test = np.moveaxis(validation_images_clean, 1, -1)

x_train_noisy = np.moveaxis(training_images_clean, 1, -1) \
                + sigma * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)

x_test_noisy = np.moveaxis(validation_images_noisy, 1, -1)

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

config = ConfigProto()
config.gpu_options.allow_growth = True
set_session(Session(config=config))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

val_pred = np.moveaxis(autoencoder.predict(np.moveaxis(validation_images_noisy, 1, -1)), -1, 1)
remaining_noise(validation_images_clean, val_pred)

prediction = np.moveaxis(autoencoder.predict(np.moveaxis(test_images_noisy, 1, -1)), -1, 1)
inout.save(prediction)
