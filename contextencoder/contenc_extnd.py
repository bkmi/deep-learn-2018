import tensorflow as tf

# mask is 1s when dropped, 0s when not dropped
reconstruction_loss = tf.reduce_mean(
    mask * tf.squared_difference(image, decoded),
    axis=[1, 2, 3]
)


# example variable creation with scope
with tf.variable_scope(name_or_scope="discriminator", reuse=reuse):
    x = tf.layers.dense(
        x,
        128,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        name='relu',
        reuse=reuse
    )

# example discrimator use with basic gan loss function
D_real = discriminator(real_batch, reuse=False)
D_fake = discriminator(fabri_batch, reuse=True)

D_loss = -tf.reduce_mean(tf.log(D_real + 1e-12) + tf.log(1. - D_fake + 1e-12), name='discriminator_loss')
G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-12), name='generator_loss')

D_solver = tf.train.AdamOptimizer().minimize(
    D_loss,
    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
    name='discriminator_solver'
)
G_solver = tf.train.AdamOptimizer().minimize(
    G_loss,
    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'),
    name='generator_solver'
)

# training example
for i in range(epochs):
    # sess.run(iterator.initializer, feed_dict={real_images: train_x})

    for j in range(train_x.shape[0] // 128):
        X_mb, _ = mnist.train.next_batch(batch_size)
        X_mb = np.reshape(X_mb, newshape=[-1, 28, 28, 1])

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={real_batch: X_mb,
                                                                 noise: sample_Z(batch_size, noise_dims)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={noise: sample_Z(batch_size, noise_dims)})