def triplet_loss(alpha):
	anchor = tf.placeholder(tf.float32, shape=input_shape)
	positive = tf.placeholder(tf.float32, shape=input_shape)
	negative = tf.placeholder(tf.float32, shape=input_shape)
	anchor_out = get_model(anchor, reuse=True)
	positive_out = get_model(positive, reuse=True)
	negative_out = get_model(negative, reuse=True)
	with tf.variable_scope('triplet_loss'):
		pos_dist = distance_metric(anchor_out, positive_out, metric='euclidean') 
		neg_dist = distance_metric(anchor_out, negative_out, metric='euclidean')
		basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
		loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
		return loss