#!/usr/bin/env python


import glob, re
import numpy as np
import sys

import matplotlib
from scipy import signal

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
font = {'family' : 'FreeSerif',
        'size'   : 18}
#plt.rc('text', usetex=True)
matplotlib.rc('font', **font)
plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125
plt.rcParams['legend.numpoints'] = 1


def line_plot(y, num_to_label, file_name, title="t-SNE Embedding of DCNN Clustering Network"):
    #x_min, x_max = np.min(x, 0), np.max(x, 0)
    #x = x * 5    
    entries = []
    x = np.asarray(range(len(y)))
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = x * 5

    plt.figure(figsize=(6, 4))
    for ii in range(y.shape[1]):
    	#print(ii)plt.figure(figsize=(4, 4))
    	entries = entries + [plt.plot(x, y[:, ii], marker='o', color = plt.cm.tab10(ii), label=num_to_label[ii])]
    plt.legend(ncol=2, loc=8, bbox_to_anchor=(0.45, -0.55))
    #plt.legend(handles=entries, loc=8,  bbox_to_anchor=(0.5, -0.3), ncol=3)
    ax = plt.gca()
    plt.ylabel('Accuracy (%)')
    plt.xlabel('1000 Iterations')
    plt.title('Accuracies Over Time')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

if __name__== '__main__':

	overall = glob.glob(sys.argv[1] + '/*confusion_matrix_epoch*.pdf')
	without_seizures = glob.glob(sys.argv[1] + '/*confusion_matrix_without_seizure_epoch*.pdf')
	with_seizures = glob.glob(sys.argv[1] + '/*confusion_matrix_with_seizure_epoch*.pdf')
	with_only_seizures = glob.glob(sys.argv[1] + '/*confusion_matrix_with_only_seizure_epoch*.pdf')

	without_seizures = [(int(re.search('epoch(.+?)_(.+?)%.pdf', str(f)).group(1)),  float(re.search('epoch(.+?)_(.+?)%.pdf', str(f)).group(2))) for f in without_seizures]

	with_seizures = [(int(re.search('epoch(.+?)_(.+?)%.pdf', str(f)).group(1)),  float(re.search('epoch(.+?)_(.+?)%.pdf', str(f)).group(2))) for f in with_seizures]

	with_only_seizures = [(int(re.search('epoch(.+?)_(.+?)%.pdf', str(f)).group(1)),  float(re.search('epoch(.+?)_(.+?)%.pdf', str(f)).group(2))) for f in with_only_seizures]

	overall = [(int(re.search('epoch(.+?)_(.+?)%.pdf', str(f)).group(1)),  float(re.search('epoch(.+?)_(.+?)%.pdf', str(f)).group(2))) for f in overall]

	results = []

	for (ii, jj), (_, kk), (_, ll), (_, mm) in zip(sorted(overall), sorted(with_seizures), sorted(without_seizures), sorted(with_only_seizures)):
		results = results + [[ii, jj, kk, ll, mm]]

	results = np.asarray(results)




	num_to_class = dict()
	 
	num_to_class[0] = 'Overall'
	num_to_class[1] = 'With Seizures'
	num_to_class[2] = 'Without Seizures'
	num_to_class[3] = 'With Only Seizures'

	x = results[:,0]
	y = results[:, 1:]

	N = 15

	w = np.ones((N, 1))/N

	y = signal.convolve2d(y, w, 'valid')

	print(y[-1,:])


	line_plot(y, num_to_class, 'averages_over_time.pdf')