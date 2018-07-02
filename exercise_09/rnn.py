import tensorflow as tf

import exercise_09,utils as utils


data_x, data_y, vali_x, vali_y, test_x = utils.load(verbose=True,
                                                    vectorize=True)

train_data = tf.data.Dataset.from_tensor_slices((data_x, data_y))
train_iter = train_data.make_one_shot_iterator()
validate_data = tf.data.Dataset.from_tensor_slices((vali_x, vali_y))
test_data = tf.data.Dataset.from_tensor_slices(data_x)

# dataset = tf.data.Dataset.range(100)
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# for i in range(100):
#   value = sess.run(next_element)
#   assert i == value