from __future__ import print_function

import datetime
import itertools
import matplotlib
import os
import re

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "FreeSerif"

import numpy as np
import os
import psutil
import random
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

curr_time = datetime.datetime.now()

loss_mem = []
loss_mem_skip = []


def norm_op(vector, axisss):
	return normalize(vector, axis=axisss, norm='l2')
	#return vector * 10e4

def plot_embedding(X, y, epoch, accuracy, num_to_label, title):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	cmap = plt.get_cmap('gist_rainbow')
	color_map = [cmap(1.*i/6) for i in range(6)]
	legend_entry = []
	for ii, c in enumerate(color_map):
		legend_entry.append(matplotlib.patches.Patch(color=c, label=num_to_label[ii]))


	plt.figure(figsize=(4.0, 4.0))
	plt.scatter(X[:,0], X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(color_map), s=2)
	plt.legend(handles=legend_entry)
	plt.xticks([]), plt.yticks([])
	plt.title(title)
	plt.savefig('./%s Results/%s_tSNE_plot_epoch%s_%.3f%%.pdf' % (curr_time, curr_time, epoch, accuracy), bbox_inches='tight')

def compute_tSNE(X, y, epoch, accuracy, num_to_label, with_seizure=None, title="t-SNE Embedding of DCNN Clustering Network"):
	tsne = TSNE(n_components=2, init='random', random_state=0)
	X_tsne = tsne.fit_transform(X)
	plot_embedding(X_tsne, y, epoch=epoch, accuracy=accuracy, num_to_label=num_to_label, title=title)
	if with_seizure is None:
		np.savez('./%s Results/%s_tSNE_plot_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, accuracy), X_tsne, y)
	elif with_seizure == True:
		np.savez('./%s Results/%s_tSNE_plot_with_seizure_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, accuracy), X_tsne, y)
	elif with_seizure == False:
		np.savez('./%s Results/%s_tSNE_plot_without_seizure_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, accuracy), X_tsne, y)

def get_loss(loss_mem, loss_mem_skip):
	plt.figure(figsize=(4.0, 4.0))
	plt.plot(loss_mem_skip, 'ro-', markersize=2)
	plt.xlabel("1000 Iterations")
	plt.ylabel("Average Loss in 1000 Iterations")
	plt.title("Iterations vs. Average Loss")
	plt.savefig('./%s Results/%s_convergence_with_skip_plot.pdf' % (curr_time, curr_time), bbox_inches='tight')

	plt.figure(figsize=(4.0, 4.0))
	plt.plot(loss_mem, 'ro-', markersize=2)
	plt.xlabel("1000 Iterations")
	plt.ylabel("Average Loss in 1000 Iterations")
	plt.title("Iterations vs. Average Loss")
	plt.savefig('./%s Results/%s_convergence_plot.pdf' % (curr_time, curr_time), bbox_inches='tight')


def plot_confusion_matrix(cm, classes, normalize=True, cmap=plt.cm.Greys, accuracy = None, epoch=None, with_seizure=None, title = "Confusion Matrix on All Data"):
	plt.figure(figsize=(4, 4))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	ax = plt.gca()
	#plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	ax.yaxis.set_label_coords(-0.1,1.03)
	h = ax.set_ylabel('True label', rotation=0, horizontalalignment='left')

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{0:.2f}'.format(cm[i, j]), horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black") 

	#plt.tight_layout()
	plt.xlabel('Predicted label')
	plt.title(title)
	#plt.show()
	if with_seizure is None:
		plt.savefig('./%s Results/%s_confusion_matrix_epoch%s_%.3f%%.pdf' % (curr_time, curr_time, epoch, accuracy), bbox_inches='tight')
	elif with_seizure == True:
		plt.savefig('./%s Results/%s_confusion_matrix_with_seizure_epoch%s_%.3f%%.pdf' % (curr_time, curr_time, epoch, accuracy), bbox_inches='tight')
	elif with_seizure == False:
		plt.savefig('./%s Results/%s_confusion_matrix_without_seizure_epoch%s_%.3f%%.pdf' % (curr_time, curr_time, epoch, accuracy), bbox_inches='tight')


class BrainNet:
	def __init__(self, input_shape=[None, 71, 125], path_to_files='/media/krishna/DATA', l2_weight=0.05, num_output=64, num_classes=6, alpha=.5, validation_size=500, learning_rate=1e-3, batch_size=100, train_epoch=5, keep_prob=None, debug=True, restore_dir=None):
		self.bckg_num = 0
		self.artf_num = 1
		self.eybl_num = 2
		self.gped_num = 3
		self.spsw_num = 4
		self.pled_num = 5
		self.path_to_files = path_to_files

		self.num_to_class = dict()

		self.num_to_class[0] = 'BCKG'
		self.num_to_class[1] = 'ARTF'
		self.num_to_class[2] = 'EYBL'
		self.num_to_class[3] = 'GPED'
		self.num_to_class[4] = 'SPSW'
		self.num_to_class[5] = 'PLED'

		self.count_of_triplets = dict()

		self.DEBUG = debug

		self.train_path = os.path.abspath(self.path_to_files + '/Train')
		self.val_path = os.path.abspath(self.path_to_files + '/Validation')

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

		if path_to_files != '/media/krishna/DATA':
			self.artf = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.artf])
			self.bckg = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.bckg])
			self.spsw = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.spsw])
			self.pled = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.pled])
			self.gped = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.gped])
			self.eybl = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.eybl])

			self.artf_val = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.artf_val])
			self.bckg_val = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.bckg_val])
			self.spsw_val = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.spsw_val])
			self.pled_val = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.pled_val])
			self.gped_val = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.gped_val])
			self.eybl_val = np.asarray([s.replace('/media/krishna/DATA', self.path_to_files) for s in self.eybl_val])

		files_with_spsw = set(['session' + re.search('session(.+?)_', a).group(1) + '_' for a in self.spsw_val])
		files_with_gped = set(['session' + re.search('session(.+?)_', a).group(1) + '_' for a in self.gped_val])
		files_with_pled = set(['session' + re.search('session(.+?)_', a).group(1) + '_' for a in self.pled_val])
		files_with_bckg = set(['session' + re.search('session(.+?)_', a).group(1) + '_' for a in self.bckg_val])
		files_with_artf = set(['session' + re.search('session(.+?)_', a).group(1) + '_' for a in self.artf_val])
		files_with_eybl = set(['session' + re.search('session(.+?)_', a).group(1) + '_' for a in self.eybl_val])

		total_set = (files_with_spsw | files_with_gped | files_with_pled | files_with_bckg | files_with_artf | files_with_eybl)
		self.files_without_seizures = total_set - files_with_spsw - files_with_pled - files_with_gped
		self.files_with_seizures = total_set - self.files_without_seizures

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
		self.inference_input = tf.placeholder(tf.float32, shape=input_shape)
		self.inference_model = self.get_model(self.inference_input, reuse=False)
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
			file.write('DCNN Clustering Network\n')
			file.write('Normalization on\n')
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

	def distance_metric(self, a, b, metric='cosine'):
		if metric == 'cosine':
			num = tf.reduce_sum(a*b, 1)
			denom = tf.sqrt(tf.reduce_sum(a*a,1))*tf.sqrt(tf.reduce_sum(b*b, 1))
			result = 1 - (self.num/self.denom)
			return result
		elif metric=='euclidean':
			return tf.reduce_sum(tf.square(tf.subtract(a, b)), 1)


	def triplet_loss(self, alpha):
		self.anchor = tf.placeholder(tf.float32, shape=self.input_shape)
		self.positive = tf.placeholder(tf.float32, shape=self.input_shape)
		self.negative = tf.placeholder(tf.float32, shape=self.input_shape)
		self.anchor_out = self.get_model(self.anchor, reuse=True)
		self.positive_out = self.get_model(self.positive, reuse=True)
		self.negative_out = self.get_model(self.negative, reuse=True)
		with tf.variable_scope('triplet_loss'):
			pos_dist = self.distance_metric(self.anchor_out, self.positive_out, metric='euclidean') 
			neg_dist = self.distance_metric(self.anchor_out, self.negative_out, metric='euclidean')
			basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
			loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
			return loss

	def get_triplets(self, size=10):
		A = []
		P = []
		N = []

		for _ in range(size):
			choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf']
			neg_choices = list(choices)
			choice = random.choice(choices)
			neg_choices.remove(choice)

			if choice == 'bckg':
				a = np.load(random.choice(self.bckg))
				p = np.load(random.choice(self.bckg))
			elif choice == 'eybl':
				a = np.load(random.choice(self.eybl))
				p = np.load(random.choice(self.eybl))
			elif choice == 'gped':
				a = np.load(random.choice(self.gped))
				p = np.load(random.choice(self.gped))
			elif choice == 'spsw':
				a = np.load(random.choice(self.spsw))
				p = np.load(random.choice(self.spsw))
			elif choice == 'pled':
				a = np.load(random.choice(self.pled))
				p = np.load(random.choice(self.pled))
			else:
				a = np.load(random.choice(self.artf))
				p = np.load(random.choice(self.artf))

			neg_choice = random.choice(neg_choices)

			if neg_choice == 'bckg':
				n = np.load(random.choice(self.bckg))
			elif neg_choice == 'eybl':
				n = np.load(random.choice(self.eybl))
			elif neg_choice == 'gped':
				n = np.load(random.choice(self.gped))
			elif neg_choice == 'spsw':
				n = np.load(random.choice(self.spsw))
			elif neg_choice == 'pled':
				n = np.load(random.choice(self.pled))
			else:
				n = np.load(random.choice(self.artf))

			key = choice + choice + neg_choice

			if key in self.count_of_triplets:
				self.count_of_triplets[key]+=1
			else:
				self.count_of_triplets[key] = 1

			a = norm_op(a, axisss=0)
			p = norm_op(p, axisss=0)
			n = norm_op(n, axisss=0)
			A.append(a)
			P.append(p)
			N.append(n)


		A = np.asarray(A)
		P = np.asarray(P)
		N = np.asarray(N)
		return A, P, N

	# End new stuff
	# 
	def simple_model(self, inputs, reuse=False):
		with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), weights_regularizer=slim.l2_regularizer(self.l2_weight), reuse=reuse):
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
			net = slim.layers.fully_connected(net, self.num_output, weights_regularizer=None, scope='output')
			return net

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

	def get_model(self, inputs, reuse=False, use_inception=True):
		if not use_inception:
			return self.simple_model(inputs, reuse=reuse)
		else:
			return self.inception_v3(inputs, reuse=reuse)

	def train_model(self, outdir=None):
		loss = self.triplet_loss(alpha=self.alpha)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.optim = self.optimizer.minimize(loss=loss)
		self.sess.run(tf.global_variables_initializer())

		count = 0
		ii = 0
		val_percentage = 0
		val_conf_matrix = 0
		epoch = -1
		while True:
			epoch += 1
			ii = 0
			count = 0
			temp_count = 0
			full_loss = 0
			while ii <= self.batch_size:
				ii += 1
				a, p, n = self.get_triplets()

				temploss = self.sess.run(loss, feed_dict={self.anchor: a, self.positive: p, self.negative: n})

				if temploss == 0:
					ii -= 1
					count += 1
					temp_count += 1
					continue

				full_loss += temploss

				if ((ii + epoch * self.batch_size) % 1000 == 0):
					loss_mem_skip.append(full_loss / (1000.0 + temp_count))
					loss_mem.append(full_loss / (1000.0))
					full_loss = 0
					temp_count = 0
					get_loss(loss_mem, loss_mem_skip)

				_, a, p, n = self.sess.run([self.optim, self.anchor_out, self.positive_out, self.negative_out], feed_dict={self.anchor: a, self.positive: p, self.negative: n})

				d1 = np.linalg.norm(p - a)
				d2 = np.linalg.norm(n - a)

				if self.DEBUG:
					print("Epoch: %2d, Iter: %7d, IterSkip: %7d, Loss: %.4f, P_Diff: %.4f, N_diff: %.4f" % (epoch, ii, count, temploss, d1, d2))
			val_percentage, val_conf_matrix = self.validate(epoch)
		self.sess.close()
		return epoch, val_percentage, val_conf_matrix

	def get_sample(self, size=1, validation=False, with_seizure=None):
		data_list = []
		class_list = []

		if not validation:
			for ii in range(0, size):
				choice = random.choice(['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf'])

				if choice == 'bckg':
					data_list.append(norm_op(np.load(random.choice(self.bckg)), axisss=0))
					class_list.append(self.bckg_num)
				elif choice == 'eybl':
					data_list.append(norm_op(np.load(random.choice(self.eybl)), axisss=0))
					class_list.append(self.eybl_num)
				elif choice == 'gped':
					data_list.append(norm_op(np.load(random.choice(self.gped)), axisss=0))
					class_list.append(self.gped_num)
				elif choice == 'spsw':
					data_list.append(norm_op(np.load(random.choice(self.spsw)), axisss=0))
					class_list.append(self.spsw_num)
				elif choice == 'pled':
					data_list.append(norm_op(np.load(random.choice(self.pled)), axisss=0))
					class_list.append(self.pled_num)
				else:
					data_list.append(norm_op(np.load(random.choice(self.artf)), axisss=0))
					class_list.append(self.artf_num)
		else:
			for ii in range(0, size):
				choice = random.choice(['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf'])

				if with_seizure is None: 
					if choice == 'bckg':
						data_list.append(norm_op(np.load(random.choice(self.bckg_val)), axisss=0))
						class_list.append(self.bckg_num)
					elif choice == 'eybl':
						data_list.append(norm_op(np.load(random.choice(self.eybl_val)), axisss=0))
						class_list.append(self.eybl_num)
					elif choice == 'gped':
						data_list.append(norm_op(np.load(random.choice(self.gped_val)), axisss=0))
						class_list.append(self.gped_num)
					elif choice == 'spsw':
						data_list.append(norm_op(np.load(random.choice(self.spsw_val)), axisss=0))
						class_list.append(self.spsw_num)
					elif choice == 'pled':
						data_list.append(norm_op(np.load(random.choice(self.pled_val)), axisss=0))
						class_list.append(self.pled_num)
					else:
						data_list.append(norm_op(np.load(random.choice(self.artf_val)), axisss=0))
						class_list.append(self.artf_num)
				elif with_seizure == True:
					success = False
					the_file = None
					class_num = None
					while not success:
						if choice == 'bckg': 
							the_file = random.choice(self.bckg_val)
							class_num = self.bckg_num
						elif choice == 'eybl':
							the_file = random.choice(self.eybl_val)
							class_num = self.eybl_num
						elif choice == 'gped':
							the_file = random.choice(self.gped_val)
							class_num = self.gped_num
						elif choice == 'spsw': 
							the_file = random.choice(self.spsw_val)
							class_num = self.spsw_num
						elif choice == 'pled':
							the_file = random.choice(self.pled_val)
							class_num = self.pled_num
						else: 
							the_filie = random.choice(self.artf_val)
							class_num = self.artf_num

						the_file_stripped = 'session' + re.search('session(.+?)_', the_file).group(1) + '_'

						if the_file_stripped in self.files_with_seizures:
							success = True

					data_list.append(norm_op(np.load(the_file), axisss=0))
					class_list.append(class_num)

				elif with_seizure == False:
					success = False
					the_file = None
					class_num = None
					while not success:
						if choice == 'bckg': 
							the_file = random.choice(self.bckg_val)
							class_num = self.bckg_num
						elif choice == 'eybl':
							the_file = random.choice(self.eybl_val)
							class_num = self.eybl_num
						elif choice == 'gped':
							the_file = random.choice(self.gped_val)
							class_num = self.gped_num
						elif choice == 'spsw': 
							the_file = random.choice(self.spsw_val)
							class_num = self.spsw_num
						elif choice == 'pled':
							the_file = random.choice(self.pled_val)
							class_num = self.pled_num
						else: 
							the_filie = random.choice(self.artf_val)
							class_num = self.artf_num

						the_file_stripped = 'session' + re.search('session(.+?)_', the_file).group(1) + '_'

						if the_file_stripped in self.files_without_seizures:
							success = True

					data_list.append(norm_op(np.load(the_file), axisss=0))
					class_list.append(class_num)
		return data_list, class_list

	def validate(self, epoch):
		inputs, classes = self.get_sample(size=100, validation=True)
		vector_inputs = self.sess.run(self.inference_model, feed_dict={self.inference_input: inputs})
		del inputs

		tempClassifier = neighbors.KNeighborsClassifier(31)
		tempClassifier.fit(vector_inputs, classes)

		# All data (Files with Seizures & Files without Seizures)

		val_inputs, val_classes = self.get_sample(size=self.validation_size)
		vector_val_inputs = self.sess.run(self.inference_model, feed_dict={self.inference_input: val_inputs})
		del val_inputs

		pred_class = tempClassifier.predict(vector_val_inputs)

		percentage = len([i for i, j in zip(val_classes, pred_class) if i == j]) * 100.0 / self.validation_size

		if self.DEBUG:
			print("Validation Results: %.3f%% of of %d correct" % (percentage, self.validation_size))

		val_classes = list(map(lambda x: self.num_to_class[x], val_classes))
		pred_class = list(map(lambda x: self.num_to_class[x], pred_class))
		class_labels = [0, 1, 2, 3, 4, 5]
		class_labels = list(map(lambda x: self.num_to_class[x], class_labels))
		conf_matrix = confusion_matrix(val_classes, pred_class, labels=class_labels)
		np.set_printoptions(precision=2)

		np.save('./%s Results/%s_confusion_matrix_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, percentage), conf_matrix)

		plot_confusion_matrix(conf_matrix, classes=class_labels, epoch=epoch, accuracy=percentage)

		compute_tSNE(vector_inputs, classes, epoch=epoch, accuracy=percentage, num_to_label=self.num_to_class)

		# Files with Seizures
		
		val_inputs, val_classes = self.get_sample(size=self.validation_size, validation=True, with_seizure=True)
		vector_val_inputs = self.sess.run(self.inference_model, feed_dict = {self.inference_model: val_inputs})
		del val_inputs


		pred_class = tempClassifier.predict(vector_val_inputs)

		percentage = len([i for i, j in zip(val_classes, pred_class) if i == j]) * 100.0 / self.validation_size

		if self.DEBUG:
			print("Seizure File Validation Results: %.3f%% of of %d correct" % (percentage, self.validation_size))

		val_classes = list(map(lambda x: self.num_to_class[x], val_classes))
		pred_class = list(map(lambda x: self.num_to_class[x], pred_class))
		class_labels = [0, 1, 2, 3, 4, 5]
		class_labels = list(map(lambda x: self.num_to_class[x], class_labels))
		conf_matrix = confusion_matrix(val_classes, pred_class, labels=class_labels)
		np.set_printoptions(precision=2)

		np.save('./%s Results/%s_confusion_matrix_with_seizure_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, percentage), conf_matrix)

		plot_confusion_matrix(conf_matrix, classes=class_labels, epoch=epoch, accuracy=percentage, with_seizure=True, title="Confusion Matrix on Files with Seizures")

		compute_tSNE(vector_val_inputs, classes, epoch=epoch, accuracy=percentage, num_to_label=self.num_to_class, title="t-SNE Embedding of DCNN Clustering Network on Files with Seizures")

		# Files without Seizures
		
		val_inputs, val_classes = self.get_sample(size=self.validation_size, validation=True, with_seizure=False)
		vector_val_inputs = self.sess.run(self.inference_model, feed_dict = {self.inference_model: val_inputs})
		del val_inputs


		pred_class = tempClassifier.predict(vector_val_inputs)

		percentage = len([i for i, j in zip(val_classes, pred_class) if i == j]) * 100.0 / self.validation_size

		if self.DEBUG:
			print("Seizure File Validation Results: %.3f%% of of %d correct" % (percentage, self.validation_size))

		val_classes = list(map(lambda x: self.num_to_class[x], val_classes))
		pred_class = list(map(lambda x: self.num_to_class[x], pred_class))
		class_labels = [0, 1, 2, 3, 4, 5]
		class_labels = list(map(lambda x: self.num_to_class[x], class_labels))
		conf_matrix = confusion_matrix(val_classes, pred_class, labels=class_labels)
		np.set_printoptions(precision=2)

		np.save('./%s Results/%s_confusion_matrix_without_seizure_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, percentage), conf_matrix)

		plot_confusion_matrix(conf_matrix, classes=class_labels, epoch=epoch, accuracy=percentage, with_seizure=False, title="Confusion Matrix on Files without Seizures")

		compute_tSNE(vector_val_inputs, classes, epoch=epoch, accuracy=percentage, num_to_label=self.num_to_class, title="t-SNE Embedding of DCNN Clustering Network on Files without Seizures")

		self.count_of_triplets = dict()

		return percentage, conf_matrix
