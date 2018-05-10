# coding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from module import *

		
class Net(object):
	def __init__(self, sess, args):
		self.sess = sess
		self.select_net = args.select_net
		if self.select_net == 'cnn_batch_norm':
			self.net_num = 0 # slim
		else:
			self.net_num = 1 # just cnn
		self.activation_func = args.activation_func
		if self.activation_func == 'relu':
			self.func_num = 0
		elif self.activation_func == 'lrelu':
			self.func_num = 1
		else:
			self.func_num = 2
		self.feature_map1 = args.feature_map1
		self.feature_map2 = args.feature_map2
		self.feature_map3 = args.feature_map3
		self.filter_size = args.filter_size
		self.pool_size = args.pool_size
		self.epoch = args.epoch
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.drop_rate = args.drop_rate
		
		if self.net_num == 0:
			self.nets = slim_net
		else:
			self.nets = cnn_net
			
		self.mnist = input_data.read_data_sets('./data/mnist/', one_hot = True)
		self._build_net()
	
		
		print('Network ready!')
		
		
	def _build_net(self):
		self.X = tf.placeholder(tf.float32, [None, 784])
		self.Y = tf.placeholder(tf.float32, [None, 10])
		self.keep_prob = tf.placeholder(tf.float32)		
		self.is_training = tf.placeholder(tf.bool)
		
		self.hypothesis = self.nets(self.X, self.func_num, self.feature_map1, self.feature_map2, 
			self.feature_map3, self.filter_size, self.pool_size, self.is_training, self.keep_prob)
		
		
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits = self.hypothesis, labels = self.Y))
		self.optimizer= tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

		self.pred_label = tf.argmax(self.hypothesis, 1)
		self.predicted = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.predicted, tf.float32))
	
		
	def train(self):
		self.init_op = tf.global_variables_initializer()
		self.sess.run(self.init_op)
		
		
		
		for step in range(self.epoch):
			self.total_cost = 0
			self.total_batch = int(self.mnist.train.num_examples/self.batch_size)
			
			for i in range(self.total_batch):
				self.batch_xs, self.batch_ys = self.mnist.train.next_batch(self.batch_size)
				
	
					
				_, self.c = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: self.batch_xs, self.Y: self.batch_ys,
					self.is_training: True,self.keep_prob: self.drop_rate})
					
				self.total_cost += self.c / self.total_batch
				print('\rNow training : {}/{}'.format(i+1, self.total_batch),end = '')
			print('\t')
			print('Epoch : {}, Cost_avg = {:4f}'.format(step, self.total_cost))	
			
			self.total_val_acc = 0
			self.total_batch = int(self.mnist.validation.num_examples/self.batch_size)
			for i in range(self.total_batch):
				self.batch_xs, self.batch_ys = self.mnist.validation.images[i*self.batch_size: (i+1)*self.batch_size], self.mnist.validation.labels[i*self.batch_size: (i+1)*self.batch_size]
				
				self.val_acc = self.sess.run(self.accuracy,feed_dict={self.X: self.batch_xs, self.Y: self.batch_ys,				
				self.is_training: False, self.keep_prob: 1})
				
		
				self.total_val_acc += self.val_acc / self.total_batch
				
			print('validation_acc : {}'.format(self.total_val_acc))
		
		self.total_train_acc = 0
		self.total_batch = int(self.mnist.train.num_examples/self.batch_size)
		for i in range(self.total_batch):	
		
			self.batch_xs, self.batch_ys = self.mnist.train.next_batch(self.batch_size)
			self.train_acc = self.sess.run(self.accuracy,feed_dict={self.X: self.batch_xs, self.Y: self.batch_ys,
				self.is_training: False, self.keep_prob: 1})
			self.total_train_acc += self.train_acc / self.total_batch
		
		print('Accuracy of dataset(train) : {}'.format(self.total_train_acc))

	def test(self):
	
		self.total_test_acc = 0
		self.total_batch = int(self.mnist.test.num_examples/self.batch_size)
		for i in range(self.total_batch):
			self.batch_xs, self.batch_ys = self.mnist.test.images[i*self.batch_size: (i+1)*self.batch_size], self.mnist.test.labels[i*self.batch_size: (i+1)*self.batch_size]
			self.test_acc = self.sess.run(self.accuracy,feed_dict={self.X: self.batch_xs, self.Y: self.batch_ys ,
				self.is_training: False, self.keep_prob: 1})
			self.total_test_acc += self.test_acc / self.total_batch

		
		print('Accuracy of dataset(test) : {}'.format(self.total_test_acc))
			