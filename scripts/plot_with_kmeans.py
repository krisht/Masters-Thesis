#!/usr/bin/env python

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

def show_plot(X, y, num_to_class, clf):
	clf.fit(X, y)
	x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
	y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

	h = .02
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	cmap = plt.get_cmap('tab10')
	color_map = [cmap(1.*i/10) for i in range(6)]
	legend_entry = []
	for ii, c in enumerate(color_map):
	    legend_entry.append(matplotlib.patches.Patch(color=c, label=num_to_class[ii]))


	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Pastel1, aspect='auto', origin='lower')

	plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap=matplotlib.colors.ListedColormap(color_map), s=40, edgecolor='k', linewidth='0.6')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.show()


if __name__=='__main__':
	a = np.load(sys.argv[1])
	X=a['arr_0']
	y=a['arr_1']

	num_to_class = dict()	 
	num_to_class[0] = 'BCKG'
	num_to_class[1] = 'ARTF'
	num_to_class[2] = 'EYBL'
	num_to_class[3] = 'GPED'
	num_to_class[4] = 'SPSW'
	num_to_class[5] = 'PLED'

	show_plot(X, y, num_to_class, KNeighborsClassifier(31))
	show_plot(X, y, num_to_class, KMeans(6))