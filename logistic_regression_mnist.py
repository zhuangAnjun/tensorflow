import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# define parameters for the model
learning_rate = 0.02
batch_size = 128
n_epochs = 30

# Step 1: read in data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

# Step 2: create placeholder for features and labels
# each image in MNIST data is of shape 28*28=784, therefore, each image is represented with a *784 tensor. there are 10 classes for each image, forresponding to digits 0~9.each label is one hot vector
X = tf.placeholder(tf.float32, [batch_size, 784], name='placeholder_X')
Y = tf.placeholder(tf.int32, [batch_size, 10], name='placeholder_Y')

# Step 3: create weight and bias
# stddev is the standard deviation of the normal distribution
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.02), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# Step 4: build model
# The model returns the logits, this logits will be passed through softmax layer
logits = tf.matmul(X, w) + b

# Step 5: define loss function
# Using cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y, name='loss')
# compute the mean over all the examples in the batch
loss = tf.reduce_mean(entropy)

# Step 6: define training operation
# Using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialization and get number of batches
    writer = tf.summary.FileWriter('./graphs/logistic_regression', sess.graph)
    # Initializer global variables
    sess.run(tf.global_variables_initializer())
    # Number of batches
    n_batches = int(mnist.train.num_examples / batch_size)
    # Step 8: start training model
    # more epochs can get better performance
    for i in range(n_epochs + 10):
        total_loss = 0

        for _ in range(n_batches):
            # Get the training batches
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            # run optimizer and loss
            _, loss_batch = sess.run([optimizer, loss], feed_dict={
                X: X_batch, Y: Y_batch})

            total_loss += loss_batch

        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    # Step 9: test the model
    preds = tf.nn.softmax(logits)
    # tf.argmax returns the index with the largest value across axes of a tensor.1 is axis
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    # tf.cast casts a tensor to a new type.
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    # Get test number of batches
    n_test_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_test_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={
                                  X: X_batch, Y: Y_batch})
        # Note: accuracy_batch is a list, total_correct_preds is int
        total_correct_preds += accuracy_batch[0]

    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))

    writer.close()
