# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:37:04 2016

@author: ryleyhiga
"""
import tensorflow as tf

# Build multilayer convolutional neural network
# Define two functions that will initialize the weights and bias terms
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1) # Initial weights are a uniform truncated normal distribution
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) # Initial bias is a uniform vector
    return tf.Variable(initial)

# Define function that will define convolutional weighted sum of stride 1    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

# Define function that will max pool a 2x2 window of the image    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], 
                          padding = "SAME")
    
def conv_nn_layer(input_tensor, window_width, window_height, input_dim, 
                  output_dim, layer_name, act=tf.nn.relu):
    """
    Defines a convolutional neural network layer
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # Define layer weights
        with tf.name_scope('weights'):
            weights = weight_variable([window_width, window_height, 
                                       input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
            
        # Define biases
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
            
        # Convolve weights on image
        with tf.name_scope('preactivation'):
            preactivate = conv2d(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            
        # Determine layer activation
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations

def fc_nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
    '''
    Defines fully-connected neural network layer
    '''
    with tf.name_scope(layer_name):
        # Define layer weights
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
            
        # Define biases
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
            
        # Convolve weights on image
        with tf.name_scope('preactivation'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            
        # Determine layer activation
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations
'''      
def pool_layer(x, name, pool = max_pool_2x2):
    with tf.name_scope(name):
        pool_output = pool(x)
    return pool_output
'''