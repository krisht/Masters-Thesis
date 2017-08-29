import random
import tensorflow as tf
from sklearn.svm import SVC

from BrainNet import BrainNet

alphas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
l2_weights = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
batch_sizes = [500, 1000, 5000, 10000, 50000, 100000]

for run in range(0, 1):
	# alpha = random.choice(alphas)Shape [-1,1,6] has negative dimensions
	# learning_rate = random.choice(learning_rates)
	# l2_weight = random.choice(l2_weights)
	# batch_size = random.choice(batch_sizes)

	batch_size = 5000
	alpha = 0.5
	learning_rate = 1e-5
	l2_weight = 0.001
	validation_size = 500

	print('Run: {:d}, Alpha: {:1.1f}, Learning Rate: {:3.2e}, L2-Weight: {:3.2e}, Batch Size: {:d}'.format(run + 1, alpha, learning_rate, l2_weight, batch_size))

	net = BrainNet(path_to_files='/media/krishna/DATA', alpha=alpha, validation_size=validation_size, learning_rate=learning_rate, l2_weight=l2_weight, batch_size=batch_size,
				   debug=True, train_epoch=20)
	blah, val_percent, val_conf_matrix = net.train_model()

	print('Validation Percentage: {:2.2f}\nConfusion Matrix:\n{}'.format(val_percent, val_conf_matrix))
