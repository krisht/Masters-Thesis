from BrainNet import BrainNet


alpha = 0.5
net = BrainNet(sess, input_shape=[None, 71, 125],
				 path_to_files='/media/krishna/My Passport/DataForUsage/labeled',
				 l2_weight=0.05, num_output=64, num_classes=6, alpha=.5, validation_size=500, learning_rate=1e-3,
				 batch_size=100, train_epoch=5, keep_prob=0.5, restore_dir=None)



alphas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
l2_weight =