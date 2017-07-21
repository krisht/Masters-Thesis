from __future__ import print_function

import datetime

curr_time = random_seed = datetime.datetime.now()
constant_seed = 42

import gc
import os
import random
import sys
import timeit
import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

#random.seed(constant_seed)

np.random.seed(constant_seed)

tf.set_random_seed(constant_seed)

global_random_state = 0
global_constant_state = 0

loss_mem = []

def get_loss(loss_mem):
    plt.figure(figsize=(15.0, 15.0))
    plt.plot(loss_mem, 'r--')
    plt.xlabel("1000 Iterations")
    plt.ylabel("Average Loss in 1000 Iterations")
    plt.title("Iterations vs. Average Loss")
    plt.savefig('./%s Results/%s_convergence_plot.png' % (curr_time, curr_time), bbox_inches='tight')

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, accuracy = None, epoch=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(15.0, 15.0))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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
	def __init__(self, input_shape=[None, 71, 125], path_to_files='/media/krishna/My Passport/DataForUsage/labeled', l2_weight=0.05, num_output=64, num_classes=6, alpha=.5, validation_size=500, learning_rate=1e-3, batch_size=100, train_epoch=5, keep_prob=0.5, debug=True, restore_dir=None):
		self.BCKG_NUM = 0
		self.ARTF_NUM = 1
		self.EYBL_NUM = 2
		self.GPED_NUM = 3
		self.SPSW_NUM = 4
		self.PLED_NUM = 5
		self.path_to_files = path_to_files

		self.count_of_triplets = dict()

		self.DEBUG = debug

		path = os.path.abspath(self.path_to_files)
		self.ARTF = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
					 'artf' in os.path.join(dp, f) and 'npz' in os.path.join(dp, f)]
		self.ARTF_VAL = self.ARTF[:len(self.ARTF) / 2]
		self.ARTF = self.ARTF[len(self.ARTF) / 2:]

		self.BCKG = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
					 'bckg' in os.path.join(dp, f) and 'npz' in os.path.join(dp, f)]
		self.BCKG_VAL = self.BCKG[:len(self.BCKG) / 2]
		self.BCKG = self.BCKG[len(self.BCKG) / 2:]

		self.SPSW = [self.path_to_files + '/session10/spsw0.npz',
					 self.path_to_files + '/session11/spsw0.npz',
					 self.path_to_files + '/session112/spsw0.npz',
					 self.path_to_files + '/session114/spsw0.npz',
					 self.path_to_files + '/session115/spsw0.npz',
					 self.path_to_files + '/session116/spsw0.npz',
					 self.path_to_files + '/session118/spsw0.npz',
					 self.path_to_files + '/session119/spsw0.npz',
					 self.path_to_files + '/session12/spsw0.npz',
					 self.path_to_files + '/session121/spsw0.npz',
					 self.path_to_files + '/session122/spsw0.npz',
					 self.path_to_files + '/session123/spsw0.npz',
					 self.path_to_files + '/session216/spsw0.npz',
					 self.path_to_files + '/session219/spsw0.npz',
					 self.path_to_files + '/session220/spsw0.npz',
					 self.path_to_files + '/session222/spsw0.npz',
					 self.path_to_files + '/session225/spsw0.npz',
					 self.path_to_files + '/session226/spsw0.npz',
					 self.path_to_files + '/session227/spsw0.npz',
					 self.path_to_files + '/session228/spsw0.npz',
					 self.path_to_files + '/session229/spsw0.npz',
					 self.path_to_files + '/session230/spsw0.npz',
					 self.path_to_files + '/session231/spsw0.npz',
					 self.path_to_files + '/session232/spsw0.npz',
					 self.path_to_files + '/session233/spsw0.npz',
					 self.path_to_files + '/session54/spsw0.npz',
					 self.path_to_files + '/session55/spsw0.npz',
					 self.path_to_files + '/session57/spsw0.npz',
					 self.path_to_files + '/session59/spsw0.npz',
					 self.path_to_files + '/session73/spsw0.npz',
					 self.path_to_files + '/session76/spsw0.npz',
					 self.path_to_files + '/session78/spsw0.npz',
					 self.path_to_files + '/session79/spsw0.npz',
					 self.path_to_files + '/session81/spsw0.npz',
					 self.path_to_files + '/session83/spsw0.npz',
					 self.path_to_files + '/session85/spsw0.npz',
					 self.path_to_files + '/session87/spsw0.npz',
					 self.path_to_files + '/session89/spsw0.npz',
					 self.path_to_files + '/session9/spsw0.npz',
					 self.path_to_files + '/session91/spsw0.npz',
					 self.path_to_files + '/session92/spsw0.npz',
					 self.path_to_files + '/session94/spsw0.npz',
					 self.path_to_files + '/session95/spsw0.npz',
					 self.path_to_files + '/session96/spsw0.npz',
					 self.path_to_files + '/session99/spsw0.npz',
					 self.path_to_files + '/session127/spsw0.npz',
					 self.path_to_files + '/session129/spsw0.npz',
					 self.path_to_files + '/session130/spsw0.npz',
					 self.path_to_files + '/session131/spsw0.npz',
					 self.path_to_files + '/session132/spsw0.npz',
					 self.path_to_files + '/session133/spsw0.npz',
					 self.path_to_files + '/session135/spsw0.npz',
					 self.path_to_files + '/session136/spsw0.npz',
					 self.path_to_files + '/session137/spsw0.npz',
					 self.path_to_files + '/session138/spsw0.npz',
					 self.path_to_files + '/session139/spsw0.npz',
					 self.path_to_files + '/session14/spsw0.npz',
					 self.path_to_files + '/session140/spsw0.npz',
					 self.path_to_files + '/session141/spsw0.npz',
					 self.path_to_files + '/session142/spsw0.npz',
					 self.path_to_files + '/session143/spsw0.npz',
					 self.path_to_files + '/session144/spsw0.npz',
					 self.path_to_files + '/session17/spsw0.npz',
					 self.path_to_files + '/session234/spsw0.npz',
					 self.path_to_files + '/session255/spsw0.npz',
					 self.path_to_files + '/session276/spsw0.npz',
					 self.path_to_files + '/session298/spsw0.npz',
					 self.path_to_files + '/session32/spsw0.npz',
					 self.path_to_files + '/session358/spsw0.npz',
					 self.path_to_files + '/session53/spsw0.npz',
					 self.path_to_files + '/session80/spsw0.npz',
					 self.path_to_files + '/session146/spsw0.npz',
					 self.path_to_files + '/session147/spsw0.npz',
					 self.path_to_files + '/session148/spsw0.npz',
					 self.path_to_files + '/session149/spsw0.npz',
					 self.path_to_files + '/session150/spsw0.npz',
					 self.path_to_files + '/session152/spsw0.npz',
					 self.path_to_files + '/session154/spsw0.npz',
					 self.path_to_files + '/session155/spsw0.npz',
					 self.path_to_files + '/session157/spsw0.npz',
					 self.path_to_files + '/session166/spsw0.npz',
					 self.path_to_files + '/session168/spsw0.npz',
					 self.path_to_files + '/session178/spsw0.npz',
					 self.path_to_files + '/session179/spsw0.npz',
					 self.path_to_files + '/session180/spsw0.npz',
					 self.path_to_files + '/session181/spsw0.npz',
					 self.path_to_files + '/session185/spsw0.npz',
					 self.path_to_files + '/session19/spsw0.npz',
					 self.path_to_files + '/session197/spsw0.npz',
					 self.path_to_files + '/session199/spsw0.npz',
					 self.path_to_files + '/session2/spsw0.npz',
					 self.path_to_files + '/session200/spsw0.npz',
					 self.path_to_files + '/session201/spsw0.npz',
					 self.path_to_files + '/session203/spsw0.npz',
					 self.path_to_files + '/session205/spsw0.npz',
					 self.path_to_files + '/session206/spsw0.npz',
					 self.path_to_files + '/session207/spsw0.npz',
					 self.path_to_files + '/session212/spsw0.npz',
					 self.path_to_files + '/session213/spsw0.npz',
					 self.path_to_files + '/session235/spsw0.npz',
					 self.path_to_files + '/session237/spsw0.npz',
					 self.path_to_files + '/session24/spsw0.npz',
					 self.path_to_files + '/session241/spsw0.npz',
					 self.path_to_files + '/session244/spsw0.npz',
					 self.path_to_files + '/session245/spsw0.npz',
					 self.path_to_files + '/session246/spsw0.npz',
					 self.path_to_files + '/session247/spsw0.npz',
					 self.path_to_files + '/session248/spsw0.npz',
					 self.path_to_files + '/session249/spsw0.npz',
					 self.path_to_files + '/session25/spsw0.npz',
					 self.path_to_files + '/session254/spsw0.npz',
					 self.path_to_files + '/session256/spsw0.npz',
					 self.path_to_files + '/session258/spsw0.npz',
					 self.path_to_files + '/session259/spsw0.npz',
					 self.path_to_files + '/session261/spsw0.npz',
					 self.path_to_files + '/session262/spsw0.npz',
					 self.path_to_files + '/session264/spsw0.npz',
					 self.path_to_files + '/session269/spsw0.npz',
					 self.path_to_files + '/session27/spsw0.npz',
					 self.path_to_files + '/session270/spsw0.npz',
					 self.path_to_files + '/session274/spsw0.npz',
					 self.path_to_files + '/session277/spsw0.npz',
					 self.path_to_files + '/session279/spsw0.npz',
					 self.path_to_files + '/session28/spsw0.npz',
					 self.path_to_files + '/session280/spsw0.npz',
					 self.path_to_files + '/session281/spsw0.npz',
					 self.path_to_files + '/session282/spsw0.npz',
					 self.path_to_files + '/session283/spsw0.npz',
					 self.path_to_files + '/session284/spsw0.npz',
					 self.path_to_files + '/session285/spsw0.npz',
					 self.path_to_files + '/session287/spsw0.npz',
					 self.path_to_files + '/session288/spsw0.npz',
					 self.path_to_files + '/session289/spsw0.npz',
					 self.path_to_files + '/session29/spsw0.npz',
					 self.path_to_files + '/session291/spsw0.npz',
					 self.path_to_files + '/session295/spsw0.npz',
					 self.path_to_files + '/session296/spsw0.npz',
					 self.path_to_files + '/session297/spsw0.npz',
					 self.path_to_files + '/session299/spsw0.npz',
					 self.path_to_files + '/session30/spsw0.npz',
					 self.path_to_files + '/session300/spsw0.npz',
					 self.path_to_files + '/session301/spsw0.npz',
					 self.path_to_files + '/session302/spsw0.npz',
					 self.path_to_files + '/session304/spsw0.npz',
					 self.path_to_files + '/session305/spsw0.npz',
					 self.path_to_files + '/session306/spsw0.npz',
					 self.path_to_files + '/session307/spsw0.npz',
					 self.path_to_files + '/session308/spsw0.npz',
					 self.path_to_files + '/session309/spsw0.npz',
					 self.path_to_files + '/session31/spsw0.npz',
					 self.path_to_files + '/session310/spsw0.npz',
					 self.path_to_files + '/session314/spsw0.npz',
					 self.path_to_files + '/session317/spsw0.npz',
					 self.path_to_files + '/session319/spsw0.npz',
					 self.path_to_files + '/session320/spsw0.npz',
					 self.path_to_files + '/session321/spsw0.npz',
					 self.path_to_files + '/session322/spsw0.npz',
					 self.path_to_files + '/session323/spsw0.npz',
					 self.path_to_files + '/session324/spsw0.npz',
					 self.path_to_files + '/session325/spsw0.npz',
					 self.path_to_files + '/session326/spsw0.npz',
					 self.path_to_files + '/session327/spsw0.npz',
					 self.path_to_files + '/session328/spsw0.npz',
					 self.path_to_files + '/session329/spsw0.npz']
		self.SPSW_VAL = [self.path_to_files + '/session33/spsw0.npz',
						 self.path_to_files + '/session331/spsw0.npz',
						 self.path_to_files + '/session332/spsw0.npz',
						 self.path_to_files + '/session333/spsw0.npz',
						 self.path_to_files + '/session334/spsw0.npz',
						 self.path_to_files + '/session335/spsw0.npz',
						 self.path_to_files + '/session34/spsw0.npz',
						 self.path_to_files + '/session359/spsw0.npz',
						 self.path_to_files + '/session36/spsw0.npz',
						 self.path_to_files + '/session360/spsw0.npz',
						 self.path_to_files + '/session363/spsw0.npz',
						 self.path_to_files + '/session364/spsw0.npz',
						 self.path_to_files + '/session365/spsw0.npz',
						 self.path_to_files + '/session369/spsw0.npz',
						 self.path_to_files + '/session371/spsw0.npz',
						 self.path_to_files + '/session376/spsw0.npz',
						 self.path_to_files + '/session39/spsw0.npz',
						 self.path_to_files + '/session46/spsw0.npz',
						 self.path_to_files + '/session48/spsw0.npz',
						 self.path_to_files + '/session49/spsw0.npz',
						 self.path_to_files + '/session50/spsw0.npz']
		self.PLED = [self.path_to_files + '/session120/pled0.npz',
					 self.path_to_files + '/session232/pled0.npz',
					 self.path_to_files + '/session233/pled0.npz',
					 self.path_to_files + '/session139/pled0.npz',
					 self.path_to_files + '/session140/pled0.npz',
					 self.path_to_files + '/session141/pled0.npz',
					 self.path_to_files + '/session181/pled0.npz',
					 self.path_to_files + '/session244/pled0.npz',
					 self.path_to_files + '/session245/pled0.npz',
					 self.path_to_files + '/session247/pled0.npz',
					 self.path_to_files + '/session248/pled0.npz',
					 self.path_to_files + '/session299/pled0.npz',
					 self.path_to_files + '/session300/pled0.npz']
		self.PLED_VAL = [self.path_to_files + '/session301/pled0.npz', self.path_to_files + '/session31/pled0.npz',
						 self.path_to_files + '/session317/pled0.npz',
						 self.path_to_files + '/session319/pled0.npz',
						 self.path_to_files + '/session320/pled0.npz',
						 self.path_to_files + '/session322/pled0.npz',
						 self.path_to_files + '/session324/pled0.npz']
		self.GPED = [self.path_to_files + '/session119/gped0.npz', self.path_to_files + '/session121/gped0.npz',
					 self.path_to_files + '/session122/gped0.npz',
					 self.path_to_files + '/session123/gped0.npz',
					 self.path_to_files + '/session125/gped0.npz',
					 self.path_to_files + '/session168/gped0.npz',
					 self.path_to_files + '/session181/gped0.npz']
		self.GPED_VAL = [self.path_to_files + '/session283/gped0.npz', self.path_to_files + '/session284/gped0.npz']
		self.EYBL = [self.path_to_files + '/session0/eybl0.npz', self.path_to_files + '/session1/eybl0.npz',
					 self.path_to_files + '/session10/eybl0.npz',
					 self.path_to_files + '/session104/eybl0.npz',
					 self.path_to_files + '/session11/eybl0.npz',
					 self.path_to_files + '/session112/eybl0.npz',
					 self.path_to_files + '/session114/eybl0.npz',
					 self.path_to_files + '/session115/eybl0.npz',
					 self.path_to_files + '/session116/eybl0.npz',
					 self.path_to_files + '/session117/eybl0.npz',
					 self.path_to_files + '/session118/eybl0.npz',
					 self.path_to_files + '/session119/eybl0.npz',
					 self.path_to_files + '/session12/eybl0.npz',
					 self.path_to_files + '/session120/eybl0.npz',
					 self.path_to_files + '/session121/eybl0.npz',
					 self.path_to_files + '/session122/eybl0.npz',
					 self.path_to_files + '/session123/eybl0.npz',
					 self.path_to_files + '/session125/eybl0.npz',
					 self.path_to_files + '/session215/eybl0.npz',
					 self.path_to_files + '/session216/eybl0.npz',
					 self.path_to_files + '/session217/eybl0.npz',
					 self.path_to_files + '/session218/eybl0.npz',
					 self.path_to_files + '/session219/eybl0.npz',
					 self.path_to_files + '/session220/eybl0.npz',
					 self.path_to_files + '/session221/eybl0.npz',
					 self.path_to_files + '/session222/eybl0.npz',
					 self.path_to_files + '/session223/eybl0.npz',
					 self.path_to_files + '/session224/eybl0.npz',
					 self.path_to_files + '/session225/eybl0.npz',
					 self.path_to_files + '/session226/eybl0.npz',
					 self.path_to_files + '/session227/eybl0.npz',
					 self.path_to_files + '/session228/eybl0.npz',
					 self.path_to_files + '/session229/eybl0.npz',
					 self.path_to_files + '/session230/eybl0.npz',
					 self.path_to_files + '/session231/eybl0.npz',
					 self.path_to_files + '/session232/eybl0.npz',
					 self.path_to_files + '/session233/eybl0.npz',
					 self.path_to_files + '/session54/eybl0.npz',
					 self.path_to_files + '/session55/eybl0.npz',
					 self.path_to_files + '/session56/eybl0.npz',
					 self.path_to_files + '/session57/eybl0.npz',
					 self.path_to_files + '/session58/eybl0.npz',
					 self.path_to_files + '/session59/eybl0.npz',
					 self.path_to_files + '/session60/eybl0.npz',
					 self.path_to_files + '/session61/eybl0.npz',
					 self.path_to_files + '/session63/eybl0.npz',
					 self.path_to_files + '/session64/eybl0.npz',
					 self.path_to_files + '/session65/eybl0.npz',
					 self.path_to_files + '/session66/eybl0.npz',
					 self.path_to_files + '/session73/eybl0.npz',
					 self.path_to_files + '/session74/eybl0.npz',
					 self.path_to_files + '/session75/eybl0.npz',
					 self.path_to_files + '/session76/eybl0.npz',
					 self.path_to_files + '/session77/eybl0.npz',
					 self.path_to_files + '/session78/eybl0.npz',
					 self.path_to_files + '/session79/eybl0.npz',
					 self.path_to_files + '/session81/eybl0.npz',
					 self.path_to_files + '/session82/eybl0.npz',
					 self.path_to_files + '/session83/eybl0.npz',
					 self.path_to_files + '/session84/eybl0.npz',
					 self.path_to_files + '/session85/eybl0.npz',
					 self.path_to_files + '/session86/eybl0.npz',
					 self.path_to_files + '/session87/eybl0.npz',
					 self.path_to_files + '/session88/eybl0.npz',
					 self.path_to_files + '/session89/eybl0.npz',
					 self.path_to_files + '/session9/eybl0.npz',
					 self.path_to_files + '/session90/eybl0.npz',
					 self.path_to_files + '/session91/eybl0.npz',
					 self.path_to_files + '/session92/eybl0.npz',
					 self.path_to_files + '/session93/eybl0.npz',
					 self.path_to_files + '/session94/eybl0.npz',
					 self.path_to_files + '/session95/eybl0.npz',
					 self.path_to_files + '/session96/eybl0.npz',
					 self.path_to_files + '/session97/eybl0.npz',
					 self.path_to_files + '/session99/eybl0.npz',
					 self.path_to_files + '/session127/eybl0.npz',
					 self.path_to_files + '/session129/eybl0.npz',
					 self.path_to_files + '/session13/eybl0.npz',
					 self.path_to_files + '/session130/eybl0.npz',
					 self.path_to_files + '/session131/eybl0.npz',
					 self.path_to_files + '/session132/eybl0.npz',
					 self.path_to_files + '/session133/eybl0.npz',
					 self.path_to_files + '/session134/eybl0.npz',
					 self.path_to_files + '/session135/eybl0.npz',
					 self.path_to_files + '/session136/eybl0.npz',
					 self.path_to_files + '/session137/eybl0.npz',
					 self.path_to_files + '/session138/eybl0.npz',
					 self.path_to_files + '/session139/eybl0.npz',
					 self.path_to_files + '/session14/eybl0.npz',
					 self.path_to_files + '/session140/eybl0.npz',
					 self.path_to_files + '/session141/eybl0.npz',
					 self.path_to_files + '/session142/eybl0.npz',
					 self.path_to_files + '/session143/eybl0.npz',
					 self.path_to_files + '/session144/eybl0.npz',
					 self.path_to_files + '/session126/eybl0.npz',
					 self.path_to_files + '/session145/eybl0.npz',
					 self.path_to_files + '/session17/eybl0.npz',
					 self.path_to_files + '/session214/eybl0.npz',
					 self.path_to_files + '/session234/eybl0.npz',
					 self.path_to_files + '/session255/eybl0.npz',
					 self.path_to_files + '/session276/eybl0.npz',
					 self.path_to_files + '/session298/eybl0.npz',
					 self.path_to_files + '/session32/eybl0.npz',
					 self.path_to_files + '/session358/eybl0.npz',
					 self.path_to_files + '/session53/eybl0.npz',
					 self.path_to_files + '/session80/eybl0.npz',
					 self.path_to_files + '/session146/eybl0.npz',
					 self.path_to_files + '/session147/eybl0.npz',
					 self.path_to_files + '/session148/eybl0.npz',
					 self.path_to_files + '/session149/eybl0.npz',
					 self.path_to_files + '/session150/eybl0.npz',
					 self.path_to_files + '/session151/eybl0.npz',
					 self.path_to_files + '/session152/eybl0.npz',
					 self.path_to_files + '/session153/eybl0.npz',
					 self.path_to_files + '/session154/eybl0.npz',
					 self.path_to_files + '/session155/eybl0.npz',
					 self.path_to_files + '/session156/eybl0.npz',
					 self.path_to_files + '/session157/eybl0.npz',
					 self.path_to_files + '/session161/eybl0.npz',
					 self.path_to_files + '/session162/eybl0.npz',
					 self.path_to_files + '/session164/eybl0.npz',
					 self.path_to_files + '/session165/eybl0.npz',
					 self.path_to_files + '/session166/eybl0.npz',
					 self.path_to_files + '/session168/eybl0.npz',
					 self.path_to_files + '/session178/eybl0.npz',
					 self.path_to_files + '/session179/eybl0.npz',
					 self.path_to_files + '/session180/eybl0.npz',
					 self.path_to_files + '/session181/eybl0.npz',
					 self.path_to_files + '/session185/eybl0.npz',
					 self.path_to_files + '/session187/eybl0.npz',
					 self.path_to_files + '/session19/eybl0.npz',
					 self.path_to_files + '/session196/eybl0.npz',
					 self.path_to_files + '/session199/eybl0.npz',
					 self.path_to_files + '/session200/eybl0.npz',
					 self.path_to_files + '/session201/eybl0.npz',
					 self.path_to_files + '/session203/eybl0.npz',
					 self.path_to_files + '/session205/eybl0.npz',
					 self.path_to_files + '/session206/eybl0.npz',
					 self.path_to_files + '/session207/eybl0.npz',
					 self.path_to_files + '/session209/eybl0.npz',
					 self.path_to_files + '/session210/eybl0.npz',
					 self.path_to_files + '/session212/eybl0.npz',
					 self.path_to_files + '/session213/eybl0.npz',
					 self.path_to_files + '/session235/eybl0.npz',
					 self.path_to_files + '/session236/eybl0.npz',
					 self.path_to_files + '/session237/eybl0.npz',
					 self.path_to_files + '/session238/eybl0.npz',
					 self.path_to_files + '/session24/eybl0.npz',
					 self.path_to_files + '/session241/eybl0.npz',
					 self.path_to_files + '/session242/eybl0.npz',
					 self.path_to_files + '/session243/eybl0.npz',
					 self.path_to_files + '/session244/eybl0.npz',
					 self.path_to_files + '/session245/eybl0.npz',
					 self.path_to_files + '/session246/eybl0.npz',
					 self.path_to_files + '/session247/eybl0.npz',
					 self.path_to_files + '/session248/eybl0.npz',
					 self.path_to_files + '/session249/eybl0.npz',
					 self.path_to_files + '/session25/eybl0.npz',
					 self.path_to_files + '/session250/eybl0.npz',
					 self.path_to_files + '/session252/eybl0.npz',
					 self.path_to_files + '/session253/eybl0.npz',
					 self.path_to_files + '/session254/eybl0.npz',
					 self.path_to_files + '/session256/eybl0.npz',
					 self.path_to_files + '/session257/eybl0.npz',
					 self.path_to_files + '/session258/eybl0.npz',
					 self.path_to_files + '/session259/eybl0.npz',
					 self.path_to_files + '/session260/eybl0.npz',
					 self.path_to_files + '/session261/eybl0.npz',
					 self.path_to_files + '/session262/eybl0.npz',
					 self.path_to_files + '/session263/eybl0.npz',
					 self.path_to_files + '/session264/eybl0.npz',
					 self.path_to_files + '/session268/eybl0.npz',
					 self.path_to_files + '/session269/eybl0.npz',
					 self.path_to_files + '/session27/eybl0.npz',
					 self.path_to_files + '/session270/eybl0.npz',
					 self.path_to_files + '/session271/eybl0.npz',
					 self.path_to_files + '/session272/eybl0.npz',
					 self.path_to_files + '/session273/eybl0.npz',
					 self.path_to_files + '/session274/eybl0.npz',
					 self.path_to_files + '/session275/eybl0.npz',
					 self.path_to_files + '/session277/eybl0.npz',
					 self.path_to_files + '/session279/eybl0.npz',
					 self.path_to_files + '/session28/eybl0.npz',
					 self.path_to_files + '/session280/eybl0.npz',
					 self.path_to_files + '/session281/eybl0.npz',
					 self.path_to_files + '/session282/eybl0.npz',
					 self.path_to_files + '/session283/eybl0.npz',
					 self.path_to_files + '/session284/eybl0.npz',
					 self.path_to_files + '/session285/eybl0.npz',
					 self.path_to_files + '/session287/eybl0.npz',
					 self.path_to_files + '/session289/eybl0.npz',
					 self.path_to_files + '/session29/eybl0.npz',
					 self.path_to_files + '/session291/eybl0.npz',
					 self.path_to_files + '/session292/eybl0.npz',
					 self.path_to_files + '/session295/eybl0.npz',
					 self.path_to_files + '/session296/eybl0.npz',
					 self.path_to_files + '/session297/eybl0.npz',
					 self.path_to_files + '/session299/eybl0.npz',
					 self.path_to_files + '/session30/eybl0.npz',
					 self.path_to_files + '/session300/eybl0.npz',
					 self.path_to_files + '/session301/eybl0.npz',
					 self.path_to_files + '/session302/eybl0.npz',
					 self.path_to_files + '/session304/eybl0.npz',
					 self.path_to_files + '/session305/eybl0.npz',
					 self.path_to_files + '/session306/eybl0.npz',
					 self.path_to_files + '/session307/eybl0.npz',
					 self.path_to_files + '/session308/eybl0.npz',
					 self.path_to_files + '/session309/eybl0.npz',
					 self.path_to_files + '/session31/eybl0.npz',
					 self.path_to_files + '/session310/eybl0.npz',
					 self.path_to_files + '/session313/eybl0.npz',
					 self.path_to_files + '/session314/eybl0.npz',
					 self.path_to_files + '/session317/eybl0.npz',
					 self.path_to_files + '/session318/eybl0.npz',
					 self.path_to_files + '/session319/eybl0.npz',
					 self.path_to_files + '/session320/eybl0.npz',
					 self.path_to_files + '/session321/eybl0.npz',
					 self.path_to_files + '/session322/eybl0.npz',
					 self.path_to_files + '/session323/eybl0.npz',
					 self.path_to_files + '/session324/eybl0.npz',
					 self.path_to_files + '/session325/eybl0.npz',
					 self.path_to_files + '/session326/eybl0.npz',
					 self.path_to_files + '/session327/eybl0.npz',
					 self.path_to_files + '/session328/eybl0.npz',
					 self.path_to_files + '/session329/eybl0.npz',
					 self.path_to_files + '/session33/eybl0.npz',
					 self.path_to_files + '/session330/eybl0.npz',
					 self.path_to_files + '/session331/eybl0.npz',
					 self.path_to_files + '/session332/eybl0.npz',
					 self.path_to_files + '/session333/eybl0.npz',
					 self.path_to_files + '/session334/eybl0.npz',
					 self.path_to_files + '/session335/eybl0.npz',
					 self.path_to_files + '/session34/eybl0.npz',
					 self.path_to_files + '/session35/eybl0.npz',
					 self.path_to_files + '/session359/eybl0.npz',
					 self.path_to_files + '/session36/eybl0.npz',
					 self.path_to_files + '/session360/eybl0.npz']
		self.EYBL_VAL = [self.path_to_files + '/session363/eybl0.npz', self.path_to_files + '/session364/eybl0.npz',
						 self.path_to_files + '/session365/eybl0.npz',
						 self.path_to_files + '/session366/eybl0.npz',
						 self.path_to_files + '/session367/eybl0.npz',
						 self.path_to_files + '/session369/eybl0.npz',
						 self.path_to_files + '/session37/eybl0.npz',
						 self.path_to_files + '/session371/eybl0.npz',
						 self.path_to_files + '/session375/eybl0.npz',
						 self.path_to_files + '/session376/eybl0.npz',
						 self.path_to_files + '/session38/eybl0.npz',
						 self.path_to_files + '/session39/eybl0.npz',
						 self.path_to_files + '/session40/eybl0.npz',
						 self.path_to_files + '/session44/eybl0.npz',
						 self.path_to_files + '/session45/eybl0.npz',
						 self.path_to_files + '/session46/eybl0.npz',
						 self.path_to_files + '/session47/eybl0.npz',
						 self.path_to_files + '/session48/eybl0.npz',
						 self.path_to_files + '/session49/eybl0.npz',
						 self.path_to_files + '/session50/eybl0.npz',
						 self.path_to_files + '/session52/eybl0.npz']

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

		if not os.path.exists('./%s Results' % curr_time):
			os.makedirs('./%s Results' % curr_time)

		with open('./%s Results/METADATA.txt' % curr_time, 'w') as file:
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

		if restore_dir is not None:
			if self.DEBUG:
				print("Loading saved data...")
			dir = tf.train.Saver()
			dir.restore(self.sess, restore_dir)
			if self.DEBUG:
				print("Finished loading saved data...")

		self.load_files(first_loading=True)

	def triplet_loss(self, alpha):
		self.anchor = tf.placeholder(tf.float32, shape=self.input_shape)
		self.positive = tf.placeholder(tf.float32, shape=self.input_shape)
		self.negative = tf.placeholder(tf.float32, shape=self.input_shape)
		self.anchor_out = self.get_model(self.anchor, reuse=True)
		self.positive_out = self.get_model(self.positive, reuse=True)
		self.negative_out = self.get_model(self.negative, reuse=True)
		with tf.variable_scope('triplet_loss'):
			pos_dist = tf.reduce_sum(tf.square(self.anchor_out - self.positive_out))
			neg_dist = tf.reduce_sum(tf.square(self.anchor_out - self.negative_out))

			basic_loss = tf.maximum(0., alpha + pos_dist - neg_dist)
			loss = tf.reduce_mean(basic_loss)
			return loss

	def load_files(self, validate=False, first_loading=False):
		gc.collect()
		start_time = timeit.default_timer()
		print("Loading new source files...")
		if not validate:
			self.bckg_file = np.load(random.choice(self.BCKG))
			self.eybl_file = np.load(random.choice(self.EYBL))
			self.artf_file = np.load(random.choice(self.ARTF))
			self.gped_file = np.load(random.choice(self.GPED))
			self.pled_file = np.load(random.choice(self.PLED))
			self.spsw_file = np.load(random.choice(self.SPSW))
		else:
			self.bckg_file = np.load(random.choice(self.BCKG_VAL))
			self.eybl_file = np.load(random.choice(self.EYBL_VAL))
			self.artf_file = np.load(random.choice(self.ARTF_VAL))
			self.gped_file = np.load(random.choice(self.GPED_VAL))
			self.pled_file = np.load(random.choice(self.PLED_VAL))
			self.spsw_file = np.load(random.choice(self.SPSW_VAL))

		self.bckg = self.bckg_file['arr_0']
		self.eybl = self.eybl_file['arr_0']
		self.artf = self.artf_file['arr_0']
		self.gped = self.gped_file['arr_0']
		self.pled = self.pled_file['arr_0']
		self.spsw = self.spsw_file['arr_0']
		print("Finished loading source files in ", timeit.default_timer() - start_time, " seconds")

	def get_triplets(self):
		# global global_random_state
		# global global_constant_state

		# global_constant_state = random.getstate()
		# if global_random_state == 0:
		# 	random.seed(random_seed)
		# else:
		# 	random.setstate(global_random_state)

		choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf']
		neg_choices = choices

		choice = random.choice(choices)

		if choice in neg_choices:
			neg_choices.remove(choice)

		if choice == 'bckg':
			ii = random.randint(0, len(self.bckg) - 1)
			a = self.bckg[ii]

			jj = random.randint(0, len(self.bckg) - 1)
			p = self.bckg[jj]

		elif choice == 'eybl':
			ii = random.randint(0, len(self.eybl) - 1)
			a = self.eybl[ii]

			jj = random.randint(0, len(self.eybl) - 1)
			p = self.eybl[jj]

		elif choice == 'gped':
			ii = random.randint(0, len(self.gped) - 1)
			a = self.gped[ii]

			jj = random.randint(0, len(self.gped) - 1)
			p = self.gped[jj]

		elif choice == 'spsw':
			ii = random.randint(0, len(self.spsw) - 1)
			a = self.spsw[ii]

			jj = random.randint(0, len(self.spsw) - 1)
			p = self.spsw[jj]

		elif choice == 'pled':
			ii = random.randint(0, len(self.pled) - 1)
			a = self.pled[ii]

			jj = random.randint(0, len(self.pled) - 1)
			p = self.pled[jj]

		else:
			ii = random.randint(0, len(self.artf) - 1)
			a = self.artf[ii]

			jj = random.randint(0, len(self.artf) - 1)
			p = self.artf[jj]

		neg_choice = random.choice(neg_choices)

		if neg_choice == 'bckg':
			ii = random.randint(0, len(self.bckg) - 1)
			n = self.bckg[ii]
		elif neg_choice == 'eybl':
			ii = random.randint(0, len(self.eybl) - 1)
			n = self.eybl[ii]
		elif neg_choice == 'gped':
			ii = random.randint(0, len(self.gped) - 1)
			n = self.gped[ii]
		elif neg_choice == 'spsw':
			ii = random.randint(0, len(self.spsw) - 1)
			n = self.spsw[ii]
		elif neg_choice == 'pled':
			ii = random.randint(0, len(self.pled) - 1)
			n = self.pled[ii]
		else:
			ii = random.randint(0, len(self.artf) - 1)
			n = self.artf[ii]

		key = choice + choice + neg_choice

		if key in self.count_of_triplets:
			self.count_of_triplets[key] = self.count_of_triplets[key] + 1
		else:
			self.count_of_triplets[key] = 1

		a = np.expand_dims(a, 0) * 10e4
		p = np.expand_dims(p, 0) * 10e4
		n = np.expand_dims(n, 0) * 10e4

		# global_random_state = random.getstate()
		# random.setstate(global_constant_state)

		return np.vstack([a, p, n])

	def get_model(self, input, reuse=False):
		with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
							weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
							weights_regularizer=slim.l2_regularizer(self.l2_weight), reuse=reuse):
			net = tf.expand_dims(input, axis=3)
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
			net = slim.layers.fully_connected(net, self.num_output, activation_fn=None, weights_regularizer=None, scope='output')
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

		for epoch in range(0, self.train_epoch):
			print("In epoch {:d}".format(epoch))
			ii = 0
			count = 0
			temp_count = 0
			full_loss = 0
			while ii <= self.batch_size:
				ii += 1
				feeder = self.get_triplets()

				anchor = feeder[0]
				anchor = np.expand_dims(anchor, 0)
				positive = feeder[1]
				positive = np.expand_dims(positive, 0)
				negative = feeder[2]
				negative = np.expand_dims(negative, 0)

				temploss = self.sess.run(loss, feed_dict={self.anchor: anchor, self.positive: positive, self.negative: negative})

				if temploss == 0:
					print(anchor[0][0][0], positive[0][0][0], negative[0][0][0])
					ii -= 1
					count += 1
					temp_count += 1
					continue

				full_loss += temploss

				if ((ii + epoch * self.batch_size) % 1000 == 0):
					loss_mem.append(full_loss/(1000 + count))
					full_loss = 0
					temp_count = 0

				_, anchor, positive, negative = self.sess.run([self.optim, self.anchor_out, self.positive_out, self.negative_out],
															  feed_dict={self.anchor: anchor, self.positive: positive,
																		 self.negative: negative})

				d1 = np.linalg.norm(positive - anchor)
				d2 = np.linalg.norm(negative - anchor)

				if self.DEBUG:
					print("Epoch: ", epoch, "Iteration:", ii, ", Loss: ", temploss, ", Positive Diff: ", d1, ", Negative diff: ", d2)
					print("Iterations skipped: ", count)
			val_percentage, val_conf_matrix = self.validate(epoch)
			if(epoch < self.train_epoch - 1):
				self.load_files()

		get_loss(loss_mem)
		self.sess.close()
		gc.collect()	
		return epoch, val_percentage, val_conf_matrix

	def get_sample(self, size=1):
		data_list = []
		class_list = []

		for ii in range(0, size):
			choice = random.choice(['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf'])

			if choice == 'bckg':
				ii = random.randint(0, len(self.bckg) - 1)
				data_list.append(self.bckg[ii])
				class_list.append(self.BCKG_NUM)

			elif choice == 'eybl':
				ii = random.randint(0, len(self.eybl) - 1)
				data_list.append(self.eybl[ii])
				class_list.append(self.EYBL_NUM)
			elif choice == 'gped':
				ii = random.randint(0, len(self.gped) - 1)
				data_list.append(self.gped[ii])
				class_list.append(self.GPED_NUM)
			elif choice == 'spsw':
				ii = random.randint(0, len(self.spsw) - 1)
				data_list.append(self.spsw[ii])
				class_list.append(self.SPSW_NUM)
			elif choice == 'pled':
				ii = random.randint(0, len(self.pled) - 1)
				data_list.append(self.pled[ii])
				class_list.append(self.PLED_NUM)
			else:
				ii = random.randint(0, len(self.artf) - 1)
				data_list.append(self.artf[ii])
				class_list.append(self.ARTF_NUM)

		return data_list, class_list

	def validate(self, epoch):

		self.load_files(True)

		if self.DEBUG:
			print("Validation files loaded!")

		inputs, classes = self.get_sample(size=1000)

		vector_inputs = self.sess.run(self.inference_model, feed_dict={self.inference_input: inputs})

		knn = neighbors.KNeighborsClassifier()
		knn.fit(vector_inputs, classes)

		val_inputs, val_classes = self.get_sample(size=self.validation_size)

		vector_val_inputs = self.sess.run(self.inference_model, feed_dict={self.inference_input: val_inputs})

		pred_class = knn.predict(vector_val_inputs)

		percentage = len([i for i, j in zip(val_classes, pred_class) if i == j]) * 100.0 / self.validation_size

		if self.DEBUG:
			print("Validation Results: %.3f%% of of %d correct" % (percentage, self.validation_size))

		class_labels = [0, 1, 2, 3, 4, 5]
		conf_matrix = confusion_matrix(val_classes, pred_class, labels=class_labels)
		np.set_printoptions(precision=2)

		plot_confusion_matrix(conf_matrix, classes=class_labels, title='Confusion Matrix', epoch=epoch, accuracy=percentage)

		plt.figure(figsize=(15.0, 15.0))

		plt.bar(range(len(list(self.count_of_triplets.keys()))), self.count_of_triplets.values(), align='center', color='b')
		plt.xticks(range(len(self.count_of_triplets)), self.count_of_triplets.keys(), rotation='vertical')
		plt.subplots_adjust(bottom=0.30)
		plt.savefig('./%s Results/%striplet_distribution_epoch%s_%.3f%%.png' % (curr_time, curr_time, epoch, percentage), bbox_inches='tight')
		self.count_of_triplets=dict()
		del inputs
		del classes
		del vector_inputs
		del knn
		del val_inputs
		del val_classes
		del pred_class

		del self.bckg_file.f
		self.bckg_file.close()
		del self.bckg_file

		del self.eybl_file.f
		self.eybl_file.close()
		del self.eybl_file

		del self.artf_file.f
		self.artf_file.close()
		del self.artf_file

		del self.gped_file.f
		self.gped_file.close()
		del self.gped_file

		del self.pled_file.f
		self.pled_file.close()
		del self.pled_file

		del self.spsw_file.f
		self.spsw_file.close()
		del self.spsw_file

		del self.bckg
		del self.eybl
		del self.artf
		del self.gped
		del self.pled
		del self.spsw
		del vector_val_inputs

		gc.collect()

		return percentage, conf_matrix

