import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from util import *


def slim_conv2d(input, maps, f_sz, num, is_training):
	fn = select_fn(num)
	batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
	slim_output = slim.conv2d(input, maps, f_sz, padding = 'SAME', activation_fn = fn, weights_initializer = 
		slim.xavier_initializer(), normalizer_fn = slim.batch_norm, normalizer_params = batch_norm_params, scope = 'slim_conv')
	return slim_output

def slim_pool(input, pool_sz):
	pool_output = slim.max_pool2d(input, pool_sz, padding = 'SAME', scope = 'slim_pool')
	return pool_output

def conv2d(input, maps, f_sz, num):
	fn = select_fn(num)
	conv_output = tf.layers.conv2d(input, maps, f_sz, padding = 'SAME', activation = fn, kernel_initializer = 
		tf.contrib.layers.xavier_initializer())
	return conv_output
	
def conv_pool(input, pool_sz):
	pool_output = tf.layers.max_pooling2d(input, pool_sz, [1, 1], padding = 'SAME')
	return pool_output
	
	
	
def slim_net(x, func_num,
			feature_map1, feature_map2, 
			feature_map3, filter_size, pool_size, is_training, keep_prob, name = 'network'):
	x = tf.reshape(x, shape = [-1, 28, 28, 1])

	with tf.variable_scope('slim_1'):
		c1 = slim_conv2d(x, feature_map1, [filter_size, filter_size], func_num, is_training)
		p1 = slim_pool(c1, [pool_size, pool_size])
		d1 = slim.dropout(p1, is_training = is_training, 
			keep_prob = keep_prob, scope = 'dropout')
		
	with tf.variable_scope('slim_2'):
		c2 = slim_conv2d(d1, feature_map2, [filter_size, filter_size], func_num, is_training)
		p2 = slim_pool(c2, [pool_size, pool_size])
		d2 = slim.dropout(p2, is_training = is_training, 
			keep_prob = keep_prob, scope = 'dropout')	
			
	with tf.variable_scope('slim_3'):
		c3 = slim_conv2d(d2, feature_map3, [filter_size, filter_size], func_num, is_training)
		p3 = slim_pool(c3, [pool_size, pool_size])
		d3 = slim.dropout(p3, is_training = is_training, 
			keep_prob = keep_prob, scope = 'dropout')
	
			
	with tf.variable_scope('slim_output'):
		net = slim.flatten(d3, scope = 'flatten_layer')
		hypothesis = slim.fully_connected(net, 10, activation_fn = None)
		return hypothesis
		

def cnn_net(x, func_num,
			feature_map1, feature_map2, 
			feature_map3, filter_size, pool_size, is_training, keep_prob, name = 'network'):
	x = tf.reshape(x, shape = [-1, 28, 28, 1])		
		
		
	with tf.variable_scope('conv_1'):
		c1 = conv2d(x, feature_map1, [filter_size, filter_size], func_num)
		p1 = conv_pool(c1, [pool_size, pool_size])
		d1 = tf.layers.dropout(p1, keep_prob, is_training)
		
	with tf.variable_scope('conv_2'):
		c2 = conv2d(d1, feature_map2, [filter_size, filter_size], func_num)
		p2 = conv_pool(c2, [pool_size, pool_size])
		d2 = tf.layers.dropout(p2, keep_prob, is_training)	

	with tf.variable_scope('conv_3'):
		c3 = conv2d(d2, feature_map3, [filter_size, filter_size], func_num)
		p3 = conv_pool(c3, [pool_size, pool_size])
		d3 = tf.layers.dropout(p3, keep_prob, is_training)	
		
	with tf.variable_scope('conv_output'):
		net = tf.contrib.layers.flatten(d3)
		hypothesis = tf.layers.dense(net, 10, activation = None)
		return hypothesis
	