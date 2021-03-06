#!/usr/bin/env python

from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
font = {'family' : 'FreeSerif',
        'size'   : 18}
plt.rc('text', usetex=True)
matplotlib.rc('font', **font)
plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125
plt.rcParams['legend.numpoints'] = 1
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier


from sklearn.neural_network import MLPClassifier
def show_plot(X, y, num_to_class, clf, save=True, title='Classifier'):
	file_name = title + ".pdf"
	clf.fit(X, y)
	x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
	y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

	h = .04
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	cmap = plt.get_cmap('tab10')
	color_map = [cmap(1.*i/10) for i in range(len(set(y)))]
	legend_entry = []
	for ii, c in enumerate(color_map):
		legend_entry.append(matplotlib.patches.Patch(color=c, label=num_to_class[ii]))


	plt.figure()
	plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Pastel1, aspect='auto', origin='lower')
	plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap=matplotlib.colors.ListedColormap(color_map), s=40, edgecolor='k', linewidth='0.6')
	plt.legend(handles=legend_entry, loc=8,  bbox_to_anchor=(0.5, -0.2), ncol=2)
	ax = plt.gca()
	ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
	ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())    
	ax.set_aspect('auto')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	if not save:
		plt.show()
	else:
		plt.savefig(file_name,  bbox_inches='tight')


if __name__=='__main__':
	a = np.load(sys.argv[1])
	X=a['arr_0']
	y=a['arr_1']
	bool_y = np.asarray([1 if x > 2 else 0 for x in list(y)])

	num_to_class = dict()	 
	num_to_class[0] = 'BCKG'
	num_to_class[1] = 'ARTF'
	num_to_class[2] = 'EYBL'
	num_to_class[3] = 'GPED'
	num_to_class[4] = 'SPSW'
	num_to_class[5] = 'PLED'


	bool_to_class = dict()
	bool_to_class[0] = 'Noise'
	bool_to_class[1] = 'Seizure'

	#show_plot(X, y, num_to_class, KNeighborsClassifier(31))
	#show_plot(X, y, num_to_class, KNeighborsClassifier(31))\
	names = ["30-NN Binary Classification", "Linear SVM Binary Classification"]
	classifiers = [KNeighborsClassifier(40), LinearSVC()]
	for name, clf in zip(names, classifiers):
		show_plot(X, bool_y, bool_to_class, clf, title=name)