#!/usr/bin/env python

from __future__ import print_function

import datetime
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import random
import sys
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
matplotlib.use('Agg')

plt.rcParams["font.family"] = "FreeSerif"

curr_time = datetime.datetime.now()
def plot_confusion_matrix(cm, classes, normalize=True, cmap=plt.cm.Greys, accuracy = None, epoch=None, with_seizure=None, file_name=None, title = "Confusion Matrix on All Data"):
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
        cm = np.nan_to_num(cm)
       # print("Normalized confusion matrix")
    else:
    	pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(cm[i, j]), horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black") 

    #plt.tight_layout()
    plt.xlabel('Predicted label')
    #plt.title(title)
    #plt.show()
    plt.savefig(file_name, bbox_inches='tight')
    #print(file_name)
    plt.close()


if __name__ == '__main__':
	folder = sys.argv[1]

	l = []
	for dirpath, dirs, files in os.walk("."):
		for f in files:
			if '.npy' in f and 'confusion' in f:
				l = l + [os.path.join(dirpath, f)]

	ii = 0
	for ii, f in enumerate(l):
		dcnnconf = np.load(f)
		dcnnconf_labels = ['BCKG', 'ARTF', 'EYBL', 'GPED', 'SPSW', 'PLED']
		file_name = f.replace('.npy', '.pdf')
		plot_confusion_matrix(dcnnconf, dcnnconf_labels, file_name=file_name)
		# sys.stdout.write("\r{0}".format((float(ii)/len(l))*100))
		# sys.stdout.flush()