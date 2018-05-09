import numpy as np
import tensorflow as tf

from pipeline import loader, saver
from sklearn.model_selection import train_test_split


def model(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28 * 28])
    hidden_layer = tf.layers.dense(inputs=input_layer,
                                   units=15,
                                   activation=tf.nn.sigmoid)
    # dropout1 = tf.layers.dropout(inputs=hidden_layer, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # hidden_layer2 = tf.layers.dense(inputs=dropout1,
    #                                units=15,
    #                                activation=tf.nn.relu)
    # dropout2 = tf.layers.dropout(inputs=hidden_layer2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    output_layer = tf.layers.dense(inputs=hidden_layer, units=10)

    # TensorFlow architecture
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=output_layer, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(output_layer, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=output_layer)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdamOptimizer()
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
    y = y.astype(np.int32)
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=model)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

    if 0:
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=X_train.shape[0],
            hooks=[logging_hook])

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


    print(eval_results)


if __name__ == '__main__':
    main()
