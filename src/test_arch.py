import tensorflow as tf
import tensorflow.contrib.slim as slim

input_shape = [None, 71, 125]
num_output = 64
l2_weight = 1e-4

def model( inputs, reuse=False, scope=''):
	end_points={}
	with tf.name_scope(scope, 'inception_v3', [inputs]):
		with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected, slim.layers.batch_norm, slim.layers.dropout], weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), weights_regularizer=slim.l2_regularizer(l2_weight), reuse=reuse):
			with slim.arg_scope([slim.layers.conv2d], stride=1, padding='VALID', reuse=reuse):
				# 299 x 299 x 3
				inputs = tf.expand_dims(inputs, dim=3)
				end_points['conv0'] = slim.layers.conv2d(inputs, 32, kernel_size=3, stride=2, scope='conv0')
				# 149 x 149 x 32
				end_points['conv1'] = slim.layers.conv2d(end_points['conv0'], 32, kernel_size=3, scope='conv1')
				# 147 x 147 x 32
				end_points['conv2'] = slim.layers.conv2d(end_points['conv1'], 64, kernel_size=3, padding='SAME', scope='conv2')
				# 147 x 147 x 64
				#end_points['pool1'] = slim.layers.max_pool2d(end_points['conv2'], kernel_size=3, stride=2, scope='pool1')
				# 73 x 73 x 64
				end_points['conv3'] = slim.layers.conv2d(end_points['conv2'], 80, kernel_size=1, scope='conv3')
				# 73 x 73 x 80.
				end_points['conv4'] = slim.layers.conv2d(end_points['conv3'], 192, kernel_size=3, scope='conv4')
				# 71 x 71 x 192.
				#end_points['pool2'] = slim.layers.max_pool2d(end_points['conv4'], kernel_size=3, stride=2, scope='pool2')
				# 35 x 35 x 192.
				net = end_points['conv4']
			# Inception blocks
			with slim.arg_scope([slim.layers.conv2d], stride=1, padding='SAME', reuse=reuse):
				# mixed: 35 x 35 x 256.
				with tf.variable_scope('mixed_35x35x256a'):
					with tf.variable_scope('branch1x1'):
						branch1x1 = slim.layers.conv2d(net, 64, kernel_size=1, scope='branch1x1/conv1')
					with tf.variable_scope('branch5x5'):
						branch5x5 = slim.layers.conv2d(net, 48, kernel_size=1, scope='branch1x1/conv2')
						branch5x5 = slim.layers.conv2d(branch5x5, 64, kernel_size=5, scope='branch1x1/conv3')
					with tf.variable_scope('branch3x3dbl'):
						branch3x3dbl = slim.layers.conv2d(net, 64, kernel_size=1, scope='branch3x3dbl/conv1')
						branch3x3dbl = slim.layers.conv2d(branch3x3dbl, 96, kernel_size=3, scope='branch3x3dbl/conv2')
						branch3x3dbl = slim.layers.conv2d(branch3x3dbl, 96, kernel_size=3, scope='branch3x3dbl/conv3')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.avg_pool2d(net, kernel_size=3, stride=1, padding='SAME', scope='branch_pool/avg_pool1')
						branch_pool = slim.layers.conv2d(branch_pool, 32, kernel_size=1, scope='branch_pool/conv1')
					net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
					end_points['mixed_35x35x256a'] = net

	return net
inference_input = tf.placeholder(tf.float32, shape=input_shape)
inference_model = model(inference_input, reuse=False)

print(inference_model)