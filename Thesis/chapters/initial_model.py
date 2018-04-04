def get_model(input, reuse=False):
	with slim.arg_scope([layers.conv2d, layers.fully_connected], weights_initializer=layers.xavier_initializer(seed=random.random(), uniform=True), weights_regularizer=slim.l2_regularizer(0.05), reuse=reuse):
		net = tf.expand_dims(input, axis=3)
		net = layers.conv2d(net, num_outputs=32, kernel_size=4, scope='conv1', trainable=True)
		net = layers.max_pool2d(net, kernel_size=3, scope='maxpool1')
		net = layers.conv2d(net, num_outputs=64, kernel_size=5, scope='conv2', trainable=True)
		net = layers.max_pool2d(net, kernel_size=3, scope='maxpool2')
		net = layers.flatten(net, scope='flatten')
		net = layers.fully_connected(net, 256, scope='fc1', trainable=True)
		net = layers.fully_connected(net, 1024, scope='fc2', trainable=True)
		net = layers.fully_connected(net, num_output, activation_fn=None, weights_regularizer=None, scope='output')
		return net
