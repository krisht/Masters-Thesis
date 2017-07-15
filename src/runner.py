from BrainNet import BrainNet
import random
import tensorflow as tf

alphas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
l2_weights = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
batch_sizes = [500, 1000, 5000, 10000, 50000, 100000]

for run in range(0, 5):
	alpha = random.choice(alphas)
	learning_rate = random.choice(learning_rates)
	l2_weight = random.choice(l2_weights)
	batch_size = random.choice(batch_sizes)

	batch_size = 5000
	alpha = 2.5
	learning_rate = 1e-5
	l2_weight = 1e-5

	print('Run: {:d}, Alpha: {:1.1f}, Learning Rate: {:3.2e}, L2-Weight: {:3.2e}, Batch Size: {:d}'.format(run + 1, alpha, learning_rate,
																										   l2_weight, batch_size))

	sess = tf.Session()

	net = BrainNet(sess, alpha=alpha, learning_rate=learning_rate, l2_weight=l2_weight, batch_size=batch_size, debug=True, train_epoch=5)
	_, val_percent, val_conf_matrix = net.train_model()

	output = 'Validation Percentage: {:2.2f}\nConfusion Matrix:\n{}'.format(val_percent, val_conf_matrix)

	del sess
	del net
	del val_percent
	del val_conf_matrix
	del alpha
	del learning_rate
	del l2_weight
	del batch_size

	print(output)
