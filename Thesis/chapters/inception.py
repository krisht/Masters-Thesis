def inception_v3(self, inputs, dropout_keep_prob=0.8, reuse=False, scope=''):
	end_points = {}
	with tf.name_scope(scope, 'inception_v3', [inputs]):
		with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected, slim.layers.batch_norm, slim.layers.dropout], weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), weights_regularizer=slim.l2_regularizer(self.l2_weight), reuse=reuse):
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
				# mixed_1: 35 x 35 x 288.
				with tf.variable_scope('mixed_35x35x288a'):
					with tf.variable_scope('branch1x1'):
						branch1x1 = slim.layers.conv2d(net, 64, kernel_size=1, scope='branch1x1/conv1')
					with tf.variable_scope('branch5x5'):
						branch5x5 = slim.layers.conv2d(net, 48, kernel_size=1, scope='branch5x5/conv1')
						branch5x5 = slim.layers.conv2d(branch5x5, 64, kernel_size=5, scope='branch5x5/conv2')
					with tf.variable_scope('branch3x3dbl'):
						branch3x3dbl = slim.layers.conv2d(net, 64, kernel_size=1, scope='branch3x3dbl/conv1')
						branch3x3dbl = slim.layers.conv2d(branch3x3dbl, 96, kernel_size=3, scope='branch3x3dbl/conv2')
						branch3x3dbl = slim.layers.conv2d(branch3x3dbl, 96, kernel_size=3, scope='branch3x3dbl/conv3')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.avg_pool2d(net, kernel_size=3, stride=1, padding='SAME', scope='branch_pool/avg_pool1')
						branch_pool = slim.layers.conv2d(branch_pool, 64, kernel_size=1, scope='branch_pool/conv1')
					net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
					end_points['mixed_35x35x288a'] = net
				# mixed_2: 35 x 35 x 288.
				with tf.variable_scope('mixed_35x35x288b'):
					with tf.variable_scope('branch1x1'):
						branch1x1 = slim.layers.conv2d(net, 64, kernel_size=1, scope='branch1x1/conv1')
					with tf.variable_scope('branch5x5'):
						branch5x5 = slim.layers.conv2d(net, 48, kernel_size=1, scope='branch5x5/conv1')
						branch5x5 = slim.layers.conv2d(branch5x5, 64, kernel_size=5, scope='branch5x5/conv2')
					with tf.variable_scope('branch3x3dbl'):
						branch3x3dbl = slim.layers.conv2d(net, 64, kernel_size=1, scope='branch3x3dbl/conv1')
						branch3x3dbl = slim.layers.conv2d(branch3x3dbl, 96, kernel_size=3, scope='branch3x3dbl/conv2')
						branch3x3dbl = slim.layers.conv2d(branch3x3dbl, 96, kernel_size=3, scope='branch3x3dbl/conv3')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.avg_pool2d(net, kernel_size=3, stride=1, padding='SAME', scope='branch_pool/avg_pool1')
						branch_pool = slim.layers.conv2d(branch_pool, 64, kernel_size=1, scope='branch_pool/conv1')
					net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
					end_points['mixed_35x35x288b'] = net
				# mixed_3: 17 x 17 x 768.
				with tf.variable_scope('mixed_17x17x768a'):
					with tf.variable_scope('branch3x3'):
						branch3x3 = slim.layers.conv2d(net, 384, kernel_size=3, stride=2, padding='VALID', scope='branch3x3/conv1')
					with tf.variable_scope('branch3x3dbl'):
						branch3x3dbl = slim.layers.conv2d(net, 64, kernel_size=1, scope='branch3x3dbl/conv1')
						branch3x3dbl = slim.layers.conv2d(branch3x3dbl, 96, kernel_size=3, scope='branch3x3dbl/conv2')
						branch3x3dbl = slim.layers.conv2d(branch3x3dbl, 96, kernel_size=3, stride=2, padding='VALID', scope='branch3x3dbl/conv3')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.max_pool2d(net, kernel_size=3, stride=2, padding='VALID', scope='branch_pool/max_pool1')
					net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])
					end_points['mixed_17x17x768a'] = net
				# mixed4: 17 x 17 x 768.
				with tf.variable_scope('mixed_17x17x768b'):
					with tf.variable_scope('branch1x1'):
						branch1x1 = slim.layers.conv2d(net, 192, kernel_size=1, scope='branch1x1/conv1')
					with tf.variable_scope('branch7x7'):
						branch7x7 = slim.layers.conv2d(net, 128, kernel_size=1, scope='branch7x7/conv1')
						branch7x7 = slim.layers.conv2d(branch7x7, 128, kernel_size=(1, 7), scope='branch7x7/conv2')
						branch7x7 = slim.layers.conv2d(branch7x7, 192, kernel_size=(7, 1), scope='branch7x7/conv3')
					with tf.variable_scope('branch7x7dbl'):
						branch7x7dbl = slim.layers.conv2d(net, 128, kernel_size=1, scope='branch7x7dbl/conv1')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 128, kernel_size=(7, 1), scope='branch7x7dbl/conv2')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 128, kernel_size=(1, 7), scope='branch7x7dbl/conv3')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 128, kernel_size=(7, 1), scope='branch7x7dbl/conv4')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 192, kernel_size=(1, 7), scope='branch7x7dbl/conv5')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.avg_pool2d(net, kernel_size=3, stride=1, padding='SAME', scope='branch_pool/avg_pool1')
						branch_pool = slim.layers.conv2d(branch_pool, 192, kernel_size=1, scope='branch_pool/conv1')
					net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
					end_points['mixed_17x17x768b'] = net
				# mixed_5: 17 x 17 x 768.
				with tf.variable_scope('mixed_17x17x768c'):
					with tf.variable_scope('branch1x1'):
						branch1x1 = slim.layers.conv2d(net, 192, kernel_size=1, scope='branch1x1/conv1')
					with tf.variable_scope('branch7x7'):
						branch7x7 = slim.layers.conv2d(net, 160, kernel_size=1, scope='branch7x7/conv1')
						branch7x7 = slim.layers.conv2d(branch7x7, 160, kernel_size=(1, 7), scope='branch7x7/conv2')
						branch7x7 = slim.layers.conv2d(branch7x7, 192, kernel_size=(7, 1), scope='branch7x7/conv3')
					with tf.variable_scope('branch7x7dbl'):
						branch7x7dbl = slim.layers.conv2d(net, 160, kernel_size=1, scope='branch7x7dbl/conv1')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 160, kernel_size=(7, 1), scope='branch7x7dbl/conv2')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 160, kernel_size=(1, 7), scope='branch7x7dbl/conv3')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 160, kernel_size=(7, 1), scope='branch7x7dbl/conv4')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 192, kernel_size=(1, 7), scope='branch7x7dbl/conv5')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.avg_pool2d(net, kernel_size=3, stride=1, padding='SAME', scope='branch_pool/avg_pool1')
						branch_pool = slim.layers.conv2d(branch_pool, 192, kernel_size=1, scope='branch_pool/conv2')
					net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
					end_points['mixed_17x17x768c'] = net
				# mixed_6: 17 x 17 x 768.
				with tf.variable_scope('mixed_17x17x768d'):
					with tf.variable_scope('branch1x1'):
						branch1x1 = slim.layers.conv2d(net, 192, kernel_size=1, scope='branch1x1/conv1')
					with tf.variable_scope('branch7x7'):
						branch7x7 = slim.layers.conv2d(net, 160, kernel_size=1, scope='branch7x7/conv1')
						branch7x7 = slim.layers.conv2d(branch7x7, 160, kernel_size=(1, 7), scope='branch7x7/conv2')
						branch7x7 = slim.layers.conv2d(branch7x7, 192, kernel_size=(7, 1), scope='branch7x7/conv3')
					with tf.variable_scope('branch7x7dbl'):
						branch7x7dbl = slim.layers.conv2d(net, 160, kernel_size=1, scope='branch7x7dbl/conv1')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 160, kernel_size=(7, 1), scope='branch7x7dbl/conv2')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 160, kernel_size=(1, 7), scope='branch7x7dbl/conv3')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 160, kernel_size=(7, 1), scope='branch7x7dbl/conv4')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 192, kernel_size=(1, 7), scope='branch7x7dbl/conv5')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.avg_pool2d(net, kernel_size=3, stride=1, padding='SAME', scope='branch_pool/avg_pool1')
						branch_pool = slim.layers.conv2d(branch_pool, 192, kernel_size=1, scope='branch_pool/conv1')
					net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
					end_points['mixed_17x17x768d'] = net
				# mixed_7: 17 x 17 x 768.
				with tf.variable_scope('mixed_17x17x768e'):
					with tf.variable_scope('branch1x1'):
						branch1x1 = slim.layers.conv2d(net, 192, kernel_size=1, scope='branch1x1/conv1')
					with tf.variable_scope('branch7x7'):
						branch7x7 = slim.layers.conv2d(net, 192, kernel_size=1, scope='branch7x7/conv1')
						branch7x7 = slim.layers.conv2d(branch7x7, 192, kernel_size=(1, 7), scope='branch7x7/conv2')
						branch7x7 = slim.layers.conv2d(branch7x7, 192, kernel_size=(7, 1), scope='branch7x7/conv3')
					with tf.variable_scope('branch7x7dbl'):
						branch7x7dbl = slim.layers.conv2d(net, 192, kernel_size=1, scope='branch7x7dbl/conv1')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 192, kernel_size=(7, 1), scope='branch7x7dbl/conv2')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 192, kernel_size=(1, 7), scope='branch7x7dbl/conv3')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 192, kernel_size=(7, 1), scope='branch7x7dbl/conv4')
						branch7x7dbl = slim.layers.conv2d(branch7x7dbl, 192, kernel_size=(1, 7), scope='branch7x7dbl/conv5')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.avg_pool2d(net, kernel_size=3, stride=1, padding='SAME', scope='branch_pool/avg_pool1')
						branch_pool = slim.layers.conv2d(branch_pool, 192, kernel_size=1, scope='branch_pool/conv1')
					net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
					end_points['mixed_17x17x768e'] = net
				# Auxiliary Head logits
				aux_logits = tf.identity(end_points['mixed_17x17x768e'])
				with tf.variable_scope('aux_logits'):
					aux_logits = slim.layers.avg_pool2d(aux_logits, kernel_size=5, stride=3, padding='VALID', scope='aux_logits/avg_pool1')
					aux_logits = slim.layers.conv2d(aux_logits, 128, kernel_size=1, scope='aux_logits/proj')
					# Shape of feature map before the final layer.
					shape = aux_logits.get_shape()
					aux_logits = slim.layers.conv2d(aux_logits, 768, shape[1:3], padding='VALID', scope='aux_logits/conv2')
					aux_logits = slim.layers.flatten(aux_logits, scope='aux_logits/flatten')
					aux_logits = slim.layers.fully_connected(aux_logits, self.num_output, activation_fn=None, scope='aux_logits/fc1')
					end_points['aux_logits'] = aux_logits

				with tf.variable_scope('mixed_17x17x1280a'):
					with tf.variable_scope('branch3x3'):
						branch3x3 = slim.layers.conv2d(net, 192, kernel_size=1, scope='branch3x3/conv1')
						branch3x3 = slim.layers.conv2d(branch3x3, 320, kernel_size=3, stride=2, padding='VALID', scope='branch3x3/conv2')
					with tf.variable_scope('branch7x7x3'):
						branch7x7x3 = slim.layers.conv2d(net, 192, kernel_size=1, scope='branch7x7x3/conv1')
						branch7x7x3 = slim.layers.conv2d(branch7x7x3, 192, kernel_size=(1, 7), scope='branch7x7x3/conv2')
						branch7x7x3 = slim.layers.conv2d(branch7x7x3, 192, kernel_size=(7, 1), scope='branch7x7x3/conv3')
						branch7x7x3 = slim.layers.conv2d(branch7x7x3, 192, kernel_size=3, stride=2, padding='VALID', scope='branch7x7x3/conv4')
					with tf.variable_scope('branch_pool'):
						branch_pool = slim.layers.max_pool2d(net, kernel_size=3, stride=2, padding='VALID', scope='branch_pool/max_pool1')
					net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
					end_points['mixed_17x17x1280a'] = net
				with tf.variable_scope('logits'):
					shape = net.get_shape()
					net = slim.layers.avg_pool2d(net, shape[1:3], stride=1, padding='VALID', scope='pool')
					end_points['prev_layer'] = net
					# 1 x 1 x 2048
					#net = slim.layers.dropout(net, dropout_keep_prob, scope='dropout')
					net = slim.layers.flatten(net, scope='flatten')
					# 2048
					logits = slim.layers.fully_connected(net, self.num_output, weights_regularizer=None, activation_fn=None, scope='logits')
					# 1000
					end_points['logits'] = logits
	return end_points['logits']
