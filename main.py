# coding: utf-8

import argparse
import os
import tensorflow as tf
tf.set_random_seed(180510)
from model import *

parser = argparse.ArgumentParser()

parser.add_argument('select_net', help ='select slim or cnn')
parser.add_argument('--activation_func', dest = 'activation_func', default = 'relu', help ='select sigmoid, relu, lrelu')
parser.add_argument('--feature_map1', dest = 'feature_map1', default = 64, help ='Number of feature map1', type = int)
parser.add_argument('--feature_map2', dest = 'feature_map2', default = 128, help ='Number of feature map2', type = int)
parser.add_argument('--feature_map3', dest = 'feature_map3', default = 256, help ='Number of feature map3', type = int)
parser.add_argument('--filter_size', dest = 'filter_size', default = 3, help ='size of filter', type = int)
parser.add_argument('--pool_size', dest = 'pool_size', default = 2, help ='size of max_pooling', type = int)
parser.add_argument('--epoch', dest = 'epoch', default =10, help ='decide epoch', type = int)
parser.add_argument('--batch_size', dest = 'batch_size', default = 50, help = 'decide batch_size', type = int)
parser.add_argument('--learning_rate', dest = 'learning_rate', default = 0.001, help = 'decide batch_size', type = float)
parser.add_argument('--drop_rate', dest = 'drop_rate', default = 0.7, help = 'decide to drop rate', type = float)


args = parser.parse_args()

# define main
def main(_):
	tfconfig = tf.ConfigProto(allow_soft_placement=True)

	with tf.Session(config=tfconfig) as sess:
		networks = Net(sess, args)
		networks.train()
		networks.test()
	
	


if __name__ == '__main__':
    tf.app.run()
