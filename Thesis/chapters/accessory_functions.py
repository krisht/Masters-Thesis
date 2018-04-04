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