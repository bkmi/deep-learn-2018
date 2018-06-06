from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout, Flatten, BatchNormalization, Dense
from keras.backend.tensorflow_backend import set_session

import numpy as np
import tensorflow as tf

from pipeline import loader, saver
from sklearn.model_selection import train_test_split

X, y, X_test = loader()
X = np.moveaxis(X, 1, 3)
X_test = np.moveaxis(X_test, 1, 3)
y = y.astype(np.int32)
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.05, random_state=42)

# Add flipped versions
X_train = np.concatenate((X_train, X_train[:, :, ::-1, :]), 0)
y_train = np.concatenate((y_train, y_train), 0)

model = Sequential()
model.add(Conv2D(input_shape=X_train[0,:,:,:].shape, filters=96, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=192, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=2))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(np.unique(y_train).shape[0], activation="softmax"))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

n_epochs = 25
batch_size = 256
callbacks_list = None
H = model.fit(X_train, y_train, validation_data=(X_validate, y_validate),
              epochs=n_epochs, batch_size=batch_size, callbacks=callbacks_list)

prediction = model.predict_classes(X_test)
saver(prediction)
