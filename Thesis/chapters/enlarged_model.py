def simple_model(inputs, reuse=False):
	with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), weights_regularizer=slim.l2_regularizer(l2_weight), reuse=reuse):
		net = tf.expand_dims(inputs, dim=3)
		net = slim.layers.conv2d(net, num_outputs=32, kernel_size=5, scope='conv1', trainable=True)
		net = slim.layers.max_pool2d(net, kernel_size=5, scope='maxpool1')
		net = slim.layers.conv2d(net, num_outputs=64, kernel_size=3, scope='conv2', trainable=True)
		net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool2')
		net = slim.layers.conv2d(net, num_outputs=128, kernel_size=2, scope='conv3', trainable=True)
		net = slim.layers.max_pool2d(net, kernel_size=2, scope='maxpool3')
		net = slim.layers.conv2d(net, num_outputs=256, kernel_size=1, scope='conv4', trainable=True)
		net = slim.layers.max_pool2d(net, kernel_size=2, scope='maxpool4')
		net = slim.layers.conv2d(net, num_outputs=1024, kernel_size=4, scope='conv5', trainable=True)
		net = slim.layers.max_pool2d(net, kernel_size=4, scope='maxpool5')
		net = slim.layers.flatten(net, scope='flatten')
		net = slim.layers.fully_connected(net, 1024, scope='fc1', trainable=True)
		net = slim.layers.fully_connected(net, 512, scope='fc2', trainable=True)
		net = slim.layers.fully_connected(net, 256, scope='fc3', trainable=True)
		net = slim.layers.fully_connected(net, num_output, weights_regularizer=None, scope='output')
		return net