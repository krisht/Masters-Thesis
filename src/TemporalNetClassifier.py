from __future__ import print_function

import datetime

curr_time = random_seed = datetime.datetime.now()
constant_seed = 42

import os
import random
import sys
import copy
import timeit
import itertools
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize


# random.seed(constant_seed)

# np.random.seed(constant_seed)

# tf.set_random_seed(constant_seed)

loss_mem = []

def get_loss(loss_mem):
    plt.figure(figsize=(20.0, 20.0))
    plt.plot(loss_mem, 'ro-')
    plt.xlabel("1000 Iterations")
    plt.ylabel("Average Loss in 1000 Iterations")
    plt.title("Iterations vs. Average Loss")
    plt.savefig('./%s Results/%s_convergence_plot.png' % (curr_time, curr_time), bbox_inches='tight')

def plot_confusion_matrix(cm, classes, normalize=True, cmap=plt.cm.Greys, accuracy = None, epoch=None):
    plt.figure(figsize=(5.0, 5.0))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./%s Results/%s_confusion_matrix_epoch%s_%.3f%%.png' % (curr_time, curr_time, epoch, accuracy), bbox_inches='tight')


class BrainNet:
	def __init__(self, input_shape=[None, 71, 125], lstm_layers=10, path_to_files='/media/krishna/My Passport/DataForUsage/labeled', l2_weight=0.05, num_output=64, num_classes=6, alpha=.5, validation_size=500, learning_rate=1e-3, batch_size=100, train_epoch=5, keep_prob=0.5, debug=True, classifier=neighbors.KNeighborsClassifier(31),restore_dir=None):
		self.bckg_num = np.expand_dims(np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32),0)
		self.artf_num = np.expand_dims(np.asarray([0, 1, 0, 0, 0, 0], dtype=np.float32),0)
		self.eybl_num = np.expand_dims(np.asarray([0, 0, 1, 0, 0, 0], dtype=np.float32),0)
		self.gped_num = np.expand_dims(np.asarray([0, 0, 0, 1, 0, 0], dtype=np.float32),0)
		self.spsw_num = np.expand_dims(np.asarray([0, 0, 0, 0, 1, 0], dtype=np.float32),0)
		self.pled_num = np.expand_dims(np.asarray([0, 0, 0, 0, 0, 1], dtype=np.float32),0)

		self.num_to_class = dict()

		self.num_to_class[str(self.bckg_num)] = 'bckg'
		self.num_to_class[str(self.artf_num)] = 'artf'
		self.num_to_class[str(self.eybl_num)] = 'eybl'
		self.num_to_class[str(self.gped_num)] = 'gped'
		self.num_to_class[str(self.spsw_num)] = 'spsw'
		self.num_to_class[str(self.pled_num)] = 'pled'
		self.num_to_class[str(self.bckg_num)[1:-1]] = 'bckg'
		self.num_to_class[str(self.artf_num)[1:-1]] = 'artf'
		self.num_to_class[str(self.eybl_num)[1:-1]] = 'eybl'
		self.num_to_class[str(self.gped_num)[1:-1]] = 'gped'
		self.num_to_class[str(self.spsw_num)[1:-1]] = 'spsw'
		self.num_to_class[str(self.pled_num)[1:-1]] = 'pled'

		self.path_to_files = path_to_files

		self.count_of_triplets = dict()

		self.DEBUG = debug

		self.train_path = os.path.abspath(self.path_to_files + '/Train')
		self.val_path = os.path.abspath(self.path_to_files + '/Validation')

		self.classifier = classifier
		self.lstm_layers = lstm_layers

		path = os.path.abspath(self.path_to_files)
		self.artf = np.load(os.path.abspath(self.train_path + '/artf_files.npy'))
		self.bckg = np.load(os.path.abspath(self.train_path + '/bckg_files.npy'))
		self.spsw = np.load(os.path.abspath(self.train_path + '/spsw_files.npy'))
		self.pled = np.load(os.path.abspath(self.train_path + '/pled_files.npy'))
		self.gped = np.load(os.path.abspath(self.train_path + '/gped_files.npy'))
		self.eybl = np.load(os.path.abspath(self.train_path + '/eybl_files.npy'))

		self.artf_val = np.load(os.path.abspath(self.val_path + '/artf_files.npy'))
		self.bckg_val = np.load(os.path.abspath(self.val_path + '/bckg_files.npy'))
		self.spsw_val = np.load(os.path.abspath(self.val_path + '/spsw_files.npy'))
		self.pled_val = np.load(os.path.abspath(self.val_path + '/pled_files.npy'))
		self.gped_val = np.load(os.path.abspath(self.val_path + '/gped_files.npy'))
		self.eybl_val = np.load(os.path.abspath(self.val_path + '/eybl_files.npy'))

		self.sess = tf.Session()
		self.num_classes = num_classes
		self.num_output = num_output
		self.input_shape = input_shape
		self.batch_size = batch_size
		self.alpha = alpha
		self.train_epoch = train_epoch
		self.learning_rate = learning_rate
		self.keep_prob = keep_prob
		self.validation_size = validation_size
		self.l2_weight = l2_weight
		self.X = tf.placeholder(tf.float32, shape=[None, 125, 71])
		self.y_hat = self.get_model(self.X, reuse=False)
		self.y = tf.placeholder(tf.float32, shape=self.y_hat.get_shape())
		if restore_dir is not None:
			if self.DEBUG:
				print("Loading saved data...")
			dir = tf.train.Saver()
			dir.restore(self.sess, restore_dir)
			if self.DEBUG:
				print("Finished loading saved data...")

		if not os.path.exists('./%s Results' % curr_time):
			os.makedirs('./%s Results' % curr_time)

		self.metadata_file = './%s Results/METADATA.txt' % curr_time

		with open(self.metadata_file, 'w') as file:
			file.write('LSTM Classifier with %d Layers\n' % (self.lstm_layers))
			file.write('Time of training: %s\n' % curr_time)
			file.write('Input shape: %s\n' % input_shape)
			file.write('Path to files: %s\n' % path_to_files)
			file.write('L2 Regularization Weight: %s\n' % l2_weight)
			file.write('Number of outputs: %s\n' % num_output)
			file.write('Number of classes: %s\n' % num_classes)
			file.write('Alpha value: %s\n' % alpha)
			file.write('Validation Size: %s\n' % validation_size)
			file.write('Learning rate: %s\n' % learning_rate)
			file.write('Batch size: %s\n' % batch_size)
			file.write('Number of Epochs: %s\n' % train_epoch)
			file.write('Dropout probability: %s\n' % keep_prob)
			file.write('Debug mode: %s\n' % debug)
			file.write('Restore directory: %s\n' % restore_dir)
			file.close()

	def triplet_loss(self, alpha):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_hat, labels=self.y))
		return loss

	def get_triplets(self):
		choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf']
		choice = random.choice(choices)

		if choice == 'bckg':
			a = np.load(random.choice(self.bckg))
			label = self.bckg_num
		elif choice == 'eybl':
			a = np.load(random.choice(self.eybl))
			label = self.eybl_num
		elif choice == 'gped':
			a = np.load(random.choice(self.gped))
			label = self.gped_num
		elif choice == 'spsw':
			a = np.load(random.choice(self.spsw))
			label = self.spsw_num
		elif choice == 'pled':
			a = np.load(random.choice(self.pled))
			label = self.pled_num
		else:
			a = np.load(random.choice(self.artf))
			label = self.artf_num
 
		a = noop(a, axis=0, norm='l2').transpose()

		a = np.expand_dims(a, 0)

		return a, label

	def get_model(self, input, reuse=False):
		with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
							weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
							weights_regularizer=slim.l2_regularizer(self.l2_weight), reuse=reuse):
			net = tf.transpose(input, [1, 0, 2])
			net = tf.reshape(net, [-1, self.input_shape[1]])
			net = slim.fully_connected(net, 150, scope='fc1')
			net = tf.split(net, self.input_shape[2], 0)
			lstm1 = tf.contrib.rnn.LSTMCell(150, forget_bias=1.0, initializer=tf.contrib.layers.xavier_initializer(uniform=True), state_is_tuple=True, reuse=reuse)
			lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm1]*self.lstm_layers, state_is_tuple=True)
			net, states = tf.contrib.rnn.static_rnn(lstm_cells, net, dtype=tf.float32)
			net = net[-1]
			net = slim.fully_connected(net, self.num_output, scope='fc2', trainable=True)
			net = slim.fully_connected(net, self.num_classes, scope='fc3', activation_fn=tf.nn.softmax, trainable=False)
			return net

	def train_model(self, outdir=None):
		loss = self.triplet_loss(alpha=self.alpha)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.optim = self.optimizer.minimize(loss=loss)
		self.sess.run(tf.global_variables_initializer())

		count = 0
		ii = 0
		val_percentage = 0
		val_conf_matrix = 0
		epoch=-1
		while True:
			epoch+=1
			ii = 0
			count = 0
			temp_count = 0
			full_loss = 0
			while ii <= self.batch_size:
				ii += 1
				a, label = self.get_triplets()

				temploss, _ = self.sess.run([loss, self.optim],feed_dict={self.X: a, self.y: label})

				full_loss += temploss

				if ((ii + epoch * self.batch_size) % 1000 == 0):
					loss_mem.append(full_loss/(1000 + temp_count))
					full_loss = 0
					temp_count = 0
					get_loss(loss_mem)

				if self.DEBUG:
					print("Epoch: %2d, Iter: %7d, Loss: %.4f"% (epoch, ii, temploss))
			val_percentage, val_conf_matrix = self.validate(epoch)

		self.sess.close()
		return epoch, val_percentage, val_conf_matrix

	def get_sample(self, size=1, validation=False):
		data_list = []
		class_list = []

		if not validation: 
			for ii in range(0, size):
				choice = random.choice(['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf'])

				if choice == 'bckg':
					data_list.append(noop(np.load(random.choice(self.bckg)), axis=0, norm='l2').transpose())
					class_list.append(self.bckg_num)
				elif choice == 'eybl':
					data_list.append(noop(np.load(random.choice(self.eybl)), axis=0, norm='l2').transpose())
					class_list.append(self.eybl_num)
				elif choice == 'gped':
					data_list.append(noop(np.load(random.choice(self.gped)), axis=0, norm='l2').transpose())
					class_list.append(self.gped_num)
				elif choice == 'spsw':
					data_list.append(noop(np.load(random.choice(self.spsw)), axis=0, norm='l2').transpose())
					class_list.append(self.spsw_num)
				elif choice == 'pled':
					data_list.append(noop(np.load(random.choice(self.pled)), axis=0, norm='l2').transpose())
					class_list.append(self.pled_num)
				else:
					data_list.append(noop(np.load(random.choice(self.artf)), axis=0, norm='l2').transpose())
					class_list.append(self.artf_num)
		else:
			for ii in range(0, size):
				choice = random.choice(['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf'])

				if choice == 'bckg':
					data_list.append(noop(np.load(random.choice(self.bckg_val)), axis=0, norm='l2').transpose())
					class_list.append(self.bckg_num)
				elif choice == 'eybl':
					data_list.append(noop(np.load(random.choice(self.eybl_val)), axis=0, norm='l2').transpose())
					class_list.append(self.eybl_num)
				elif choice == 'gped':
					data_list.append(noop(np.load(random.choice(self.gped_val)), axis=0, norm='l2').transpose())
					class_list.append(self.gped_num)
				elif choice == 'spsw':
					data_list.append(noop(np.load(random.choice(self.spsw_val)), axis=0, norm='l2').transpose())
					class_list.append(self.spsw_num)
				elif choice == 'pled':
					data_list.append(noop(np.load(random.choice(self.pled_val)), axis=0, norm='l2').transpose())
					class_list.append(self.pled_num)
				else:
					data_list.append(noop(np.load(random.choice(self.artf_val)), axis=0, norm='l2').transpose())
					class_list.append(self.artf_num)		

		return data_list, np.squeeze(class_list)

	def validate(self, epoch):

		val_inputs, val_classes = self.get_sample(size=self.validation_size, validation=True)

		accuracy = self.sess.run(self.y_hat, feed_dict={self.X: val_inputs, self.y: val_classes})

		onehot_acc = np.zeros_like(accuracy)
		onehot_acc[np.arange(len(accuracy)), accuracy.argmax(1)] = 1

		percentage = len([i for i, j in zip(onehot_acc, val_classes) if (i==j).all()]) * 100.0 / self.validation_size

		if self.DEBUG:
			print("Validation Results: %.3f%% of of %d correct" % (percentage, self.validation_size))

		val_classes = list(map(lambda x : self.num_to_class[str(x)], val_classes))
		pred_class = list(map(lambda x : self.num_to_class[str(x)], onehot_acc))
		class_labels = [self.bckg_num, self.artf_num, self.eybl_num, self.gped_num, self.spsw_num, self.pled_num]
		class_labels = list(map(lambda x : self.num_to_class[str(x)], class_labels))
		conf_matrix = confusion_matrix(val_classes, pred_class, labels=class_labels)
		np.set_printoptions(precision=2)

		plot_confusion_matrix(conf_matrix, classes=class_labels, title='Confusion Matrix', epoch=epoch, accuracy=percentage)

		return percentage, conf_matrix

def noop(vector, axis, norm):
	return vector*10e4