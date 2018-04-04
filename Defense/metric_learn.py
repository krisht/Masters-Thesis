from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
font = {'family' : 'FreeSerif',
        'size'   : 18}
plt.rc('text', usetex=True)
matplotlib.rc('font', **font)
plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125
plt.rcParams['legend.numpoints'] = 1
from numpy import sqrt
X, y= make_blobs(n_samples=300, centers = [[0.1, 0.1], [0.0, 0.0], [-0.1, -0.1]], cluster_std = sqrt(2.5))
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel('x')
plt.ylabel('y')
plt.title("Example of Metric Learning")
# plt.gca().axes.set_xticklabels([])
# plt.gca().axes.set_yticklabels([])
plt.savefig('plot1.pdf', bbox_inches='tight')

X, y= make_blobs(n_samples=300, centers = [[2, 2], [0.0, 0.0], [-2, -2]], cluster_std = sqrt(2.5))
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel('x')
plt.ylabel('y')
plt.title("Example of Metric Learning")
# plt.gca().axes.set_xticklabels([])
# plt.gca().axes.set_yticklabels([])
plt.savefig('plot2.pdf', bbox_inches='tight')

X, y= make_blobs(n_samples=300, centers = [[5, 5], [0.0, 0.0], [-5, -5]], cluster_std = sqrt(2.5))
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel('x')
plt.ylabel('y')
plt.title("Example of Metric Learning")
# plt.gca().axes.set_xticklabels([])
# plt.gca().axes.set_yticklabels([])
plt.savefig('plot3.pdf', bbox_inches='tight')