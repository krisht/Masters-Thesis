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

	def get_model(self, inputs, reuse=False, use_inception=True):
		if not use_inception:
			return self.simple_model(inputs, reuse=reuse)
		else:
			return self.inception_v3(inputs, reuse=reuse)


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
					the_file = ''
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
							the_file = random.choice(self.artf_val)
							class_num = self.artf_num

						print(the_file)

						the_file_stripped = 'session' + re.search('session(.+?)_', str(the_file)).group(1) + '_'

						if the_file_stripped in self.files_with_seizures:
							success = True

					data_list.append(norm_op(np.load(str(the_file)), axisss=0))
					class_list.append(class_num)

				elif with_seizure == False:
					success = False
					the_file = ''
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
							the_file = random.choice(self.artf_val)
							class_num = self.artf_num

						print(the_file)

						the_file_stripped = 'session' + re.search('session(.+?)_', str(the_file)).group(1) + '_'

						if the_file_stripped in self.files_without_seizures:
							success = True

					data_list.append(norm_op(np.load(str(the_file)), axisss=0))
					class_list.append(class_num)
		return data_list, class_list

