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
import re
from tqdm import tqdm
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from skimage.measure import block_reduce
matplotlib.use('Agg')

plt.rcParams["font.family"] = "FreeSerif"

curr_time = datetime.datetime.now()
def plot_confusion_matrix(cm, classes, normalize=True, cmap=plt.cm.Greys, size=(4,4), file_name=None):
    plt.figure(figsize=size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    ax = plt.gca()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    ax.yaxis.set_label_coords(-0.1,1.03)
    h = ax.set_ylabel('True label', rotation=0, horizontalalignment='left')

    accuracy = np.trace(cm, dtype=np.float32)*100/np.sum(cm, dtype=np.float32)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    else:
    	pass

    file_name = re.sub("\d+\.\d+%", "%.3f%%" % accuracy, file_name)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(cm[i, j]), horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black") 

    plt.xlabel('Predicted label')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
	folder = sys.argv[1]

	l = []
	for dirpath, dirs, files in os.walk("."):
		for f in files:
			if '.npy' in f and 'confusion' in f:
				l = l + [os.path.join(dirpath, f)]

	ii = 0
	for f in tqdm(l):
		dcnnconf = np.load(f)
		dcnnconf_labels = ['BCKG', 'ARTF', 'EYBL', 'GPED', 'SPSW', 'PLED']
		file_name = f.replace('.npy', '.pdf')
		plot_confusion_matrix(dcnnconf, dcnnconf_labels, file_name=file_name)
		dcnnconf_labels = ['Noise', 'Seizure']
		file_name = f.replace('.npy', '_pooled.pdf')
		dcnnconf = block_reduce(dcnnconf, (3,3), np.sum)
		plot_confusion_matrix(dcnnconf, dcnnconf_labels, file_name=file_name, size=(2,2))