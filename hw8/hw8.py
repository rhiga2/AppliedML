# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:28:30 2016

@author: ryleyhiga
"""

import tensorflow as tf

# Build a simple softnax model in tensorflow
# Import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Initialize softmax parameters
# Digit images are 28x28 pixels
# There are 10 classifications of digits 0-9
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Implement softmax
y = tf.nn.softmax(tf.matmul(x, W)+b) # Estimate labels

# Implement cross-entropy cost-function
y_ = tf.placeholder(tf.float32, [None, 10]) # Actual training labels 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

# Use gradient optimizer to optimize training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Launch model in session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict = {x : mnist.test.images, y_ : mnist.test.labels}))    