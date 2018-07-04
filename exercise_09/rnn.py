import exercise_09.utils as utils

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D


data_x, data_y, vali_x, vali_y, test_x = utils.load(verbose=True,
                                                    vectorize=True)

max_string_length = max([data_x.shape[1], vali_x.shape[1], test_x.shape[1]])
count_features = data_x.shape[2]
data_x, vali_x, test_x = map(lambda x: sequence.pad_sequences(x, maxlen=max_string_length),
                             [data_x, vali_x, test_x])
count_possible_labels = data_y.shape[1]

model = Sequential()
model.add(Conv1D(filters=32,
                 kernel_size=3,
                 padding='same',
                 activation='relu',
                 input_shape=(max_string_length, count_features)))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(count_possible_labels,
               return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(count_possible_labels))
model.add(Dropout(0.2))
model.add(Dense(count_possible_labels,
                activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
model.fit(data_x, data_y, epochs=30, batch_size=32)

scores = model.evaluate(vali_x, vali_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
