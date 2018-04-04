#!/usr/bin/env python

import numpy as np
import matplotlib

import glob
import os
import sys

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
font = {'family' : 'FreeSerif',
        'size'   : 18}
#plt.rc('text', usetex=True)
matplotlib.rc('font', **font)
plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125
plt.rcParams['legend.numpoints'] = 1

def line_plot(x, y, num_to_label, file_name, title="t-SNE Embedding of DCNN Clustering Network"):
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = x * 5    
    entries = []
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
    plt.show()
    plt.close()


num_to_class = dict()
 
num_to_class[0] = 'Overall'
num_to_class[1] = 'With Seizures'
num_to_class[2] = 'Without Seizures'
num_to_class[3] = 'With Only Seizures'
num_to_class[4] = '5'
if __name__ == '__main__':
	file = 'over_iterations.txt'

	data = np.genfromtxt(file, delimiter=',')[1:,]

	x = np.asarray(data[:,0])
	y = np.asarray(data[:,1:5])
	#print(y)

	line_plot(x, y, num_to_class, 'averages_over_time.pdf')
