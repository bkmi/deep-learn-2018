import utils as utils
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import adam

from numpy.random import seed
from tensorflow import set_random_seed

seed(42)
set_random_seed(42)

data_x, data_y, vali_x, vali_y, test_x = utils.load(verbose=True,
                                                    vectorize=True)

max_string_length = max([data_x.shape[1], vali_x.shape[1], test_x.shape[1]])
count_features = data_x.shape[2]
data_x, vali_x, test_x = map(lambda x: sequence.pad_sequences(x, maxlen=max_string_length),
                             [data_x, vali_x, test_x])
count_possible_labels = data_y.shape[1]

model = Sequential()
model.add(Conv1D(filters=64,
                 kernel_size=100,
                 padding='same',
                 activation='relu',
                 input_shape=(max_string_length, count_features)))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=32,
                 kernel_size=50,
                 padding='same',
                 activation='relu',
                 input_shape=(max_string_length, count_features)))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(LSTM(count_possible_labels*2,
               return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(count_possible_labels))
model.add(Dropout(0.5))
model.add(Dense(count_possible_labels,
                activation='softmax'))


opt = adam(lr=.00085,
           beta_1=0.99,
           beta_2=0.9999)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# early_stop = EarlyStopping(min_delta=0.01,
#                         patience=5)
print(model.summary())
model.fit(data_x,
          data_y,
          epochs=60,
          batch_size=32,
          validation_data=(vali_x, vali_y)) # ,
#          callbacks=[early_stop])

scores = model.predict_classes(test_x)
print(scores)
np.save('prediction.npy', scores)
