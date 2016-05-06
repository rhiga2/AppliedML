"""
Created on Sat Apr 23 16:33:00 2016

@author: ryleyhiga
"""
import tensorflow as tf
from mnist_functions import *

# Make session interactive
sess = tf.InteractiveSession()
logdir = '/tmp/mnist_logs1'
if tf.gfile.Exists(logdir):
    tf.gfile.DeleteRecursively(logdir)
tf.gfile.MakeDirs(logdir)

# Import mnist dataset for classifying digits
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# Initialize placeholder variables for data x and actual labels y_
with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.float32, shape = [None, 784]) # There are 28x28 pixels => 284 features
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.image_summary('input', x_image, 10)
    y_ = tf.placeholder(tf.float32, shape = [None, 10]) # Ten possible digit classifications 0-9
    keep_prob = tf.placeholder(tf.float32) # Probability of keeping a unit during training
    tf.scalar_summary('dropout_keep_probability', keep_prob)

# Build multilayer convolutional neural network

# Describe 1st convolutional layer    
h_conv1 = conv_nn_layer(x_image, 5, 5, 1, 32, 'hidden1')
with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = conv_nn_layer(h_pool1, 5, 5, 32, 64, 'hidden2')
with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

# Describe 1st fully-connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = fc_nn_layer(h_pool2_flat, 7*7*64, 1024, 'hidden3') 

# Dropout some units during training
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Add one softmax layer at the end
y_conv = fc_nn_layer(h_fc1_drop, 1024, 10, 'output', act = tf.nn.softmax)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices = [1]))
    tf.scalar_summary('cross_entropy', cross_entropy)
    
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_predictions'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

# Create summaries for the neural network    
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(logdir + '/train', sess.graph)

tf.initialize_all_variables().run()
for i in range(2000):
    batch = mnist.train.next_batch(50)
    fdict={x : batch[0], y_ : batch[1], keep_prob : 0.9}
    if i%100 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=fdict)
        train_writer.add_summary(summary, i)
        print("step %d, training accuracy %g" % (i, acc))
    train_step.run(feed_dict = fdict)
        
test_accuracy =  accuracy.eval(feed_dict={
    x : mnist.test.images, y_ : mnist.test.labels, keep_prob : 1.0})
print('test_accuracy %d' % test_accuracy)