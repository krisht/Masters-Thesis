import random
import tensorflow as tf
from sklearn.svm import SVC
from BrainNet import BrainNet

for run in range(0, 1):
	batch_size = 5000
	alpha = 0.5
	learning_rate = 0.0001
	l2_weight = 0.001
	validation_size = 500

	print('Run: {:d}, Alpha: {:1.1f}, Learning Rate: {:3.2e}, L2-Weight: {:3.2e}, Batch Size: {:d}'.format(run + 1, alpha, learning_rate, l2_weight, batch_size))
	#path_to_files='/home/krishna/data',
	net = BrainNet(path_to_files='/home/krishna/data' alpha=alpha, validation_size=validation_size learning_rate=learning_rate, l2_weight=l2_weight batch_size=batch_size, debug=True, train_epoch=20)
	_, val_percent, val_conf_matrix = net.train_model()

	print('Validation Percentage: {:2.2f}\nConfusion Matrix:\n{}'.format(val_percent, val_conf_matrix))
