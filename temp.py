import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

for ff in ['bckg', 'artf', 'eybl', 'gped', 'spsw', 'pled']:

	path_to_files = '/media/krishna/DATA'
	gped = np.load(os.path.abspath(path_to_files + '/Train/'+ ff + '_files.npy'))



	data = np.load(random.choice(gped))

	print((data.shape[0]*2, data.shape[1]*2))

	from PIL import Image

	img = Image.fromarray(data, 'F')
	img = img.convert('P')
	img.show()
	img.save(ff + '_STFT.pdf')
