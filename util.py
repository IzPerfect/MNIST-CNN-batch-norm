import tensorflow as tf
import numpy as np


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def select_fn(num):
	if num == 0:
		activation_fn = tf.nn.relu
	elif num == 1:
		activation_fn = tf.sigmoid
	else:
		activation_fn = lrelu
		
	return activation_fn

	


