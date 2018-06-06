import numpy as np
import tensorflow as tf

from pipeline import loader, saver
from sklearn.model_selection import train_test_split
from pathlib import Path


def model(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    norm1 = tf.nn.local_response_normalization(pool1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=norm1,
        filters=96,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    norm2 = tf.nn.local_response_normalization(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

    if 0:
        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
        dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout2, units=3)
    else:
        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
        dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)

        # Logits Layer
        logits = tf.layers.dense(inputs=dense2, units=3)

    # TensorFlow architecture
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




def main():
    X, y, X_test = loader()
    X = np.moveaxis(X, 1, 3)
    # X = np.rot90(X, 0, (1,2))
    y = y.astype(np.int32)
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)

    # Restrict GPU memory
    config = tf.ConfigProto() # device_count = {'GPU': 0}
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True

    directory = Path('./estimator/')
    # if directory.exists():
    #     directory.unlink()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=model,
                                              # model_dir=directory.resolve(),
                                              config=tf.estimator.RunConfig(session_config=config))

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

    if 1:
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train,
            batch_size=50,
            num_epochs=None,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=(X_train.shape[0] / 50) * 10,
            hooks=[logging_hook])

        train_results_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train,
            num_epochs=1,
            shuffle=False)
        train_results = mnist_classifier.evaluate(input_fn=train_results_input_fn)

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_validate},
            y=y_validate,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    else:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            y=y,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=X.shape[0],
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_test},
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.predict(input_fn=eval_input_fn)
        out = np.array([i['classes'] for i in eval_results])
        saver(out)

    print(train_results)
    print(eval_results)


if __name__ == '__main__':
    main()