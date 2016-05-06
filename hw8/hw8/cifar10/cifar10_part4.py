# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:38:42 2016

@author: ryleyhiga

MY CODE FOR PART 4
"""
import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10
from tensorflow.models.image.cifar10 import cifar10_input

NUM_CLASSES = cifar10_input.NUM_CLASSES

def batch_normalize(name, x, depth, axes):
  with tf.name_scope(name):
    # Calculate mean and variance of activations
    mean, variance = tf.nn.moments(x, axes)
    # Define gamma
    with tf.name_scope('gamma'):
      scale = tf.Variable(tf.constant(1.0, shape = [depth]))
    # Define beta
    with tf.name_scope('beta'):
      offset = tf.Variable(tf.constant(0.0, shape = [depth]))
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 0.01, name = "batch_norm")
    

def myinference(images):
  # Input tensor is 128x24x24x3
  with tf.variable_scope('conv1') as scope:
    kernel = cifar10._variable_with_weight_decay('weights', shape=[5, 5, 3, 32],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = cifar10._variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    cifar10._activation_summary(conv1)
    
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    
  # Tensor conv1 is 128x24x24x32
  # Perform normalization on 1st convolutional layer.
  norm1 = batch_normalize("norm1", pool1, 32, [0,1,2])
  
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = cifar10._variable_with_weight_decay('weights', shape=[5, 5, 32, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = cifar10._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    cifar10._activation_summary(conv2)
                    
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  norm2 = batch_normalize("norm2", pool2, 64, [0,1,2])
                    
  
  # conv3                  
  with tf.variable_scope('conv3') as scope:
    kernel = cifar10._variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = cifar10._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    cifar10._activation_summary(conv3)
    
  norm3 = batch_normalize("norm3", conv3, 64, [0,1,2])
                    
  with tf.variable_scope('local4') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(norm3, [128, -1])
    dim = reshape.get_shape()[1].value
    weights = cifar10._variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = cifar10._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    cifar10._activation_summary(local4)
    
  with tf.variable_scope('local5') as scope:
    weights = cifar10._variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = cifar10._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
    cifar10._activation_summary(local5)
    
  with tf.variable_scope('softmax_linear') as scope:
    weights = cifar10._variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = cifar10._variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local5, weights), biases, name=scope.name)
    cifar10._activation_summary(softmax_linear)

  return softmax_linear