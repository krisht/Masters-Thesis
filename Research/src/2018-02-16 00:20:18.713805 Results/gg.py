import imageio
import glob
import re

overall = glob.glob('./*confusion_matrix_epoch*_pooled.png')
print(overall)
overall = sorted([(int(re.search('epoch(.+?)_(.+?)%_pooled.png', str(f)).group(1)), str(f)) for f in overall if re.search('epoch(.+?)_(.+?)%_pooled.png', str(f))])
print(overall)

images = []
for (_, f) in overall:
    images.append(imageio.imread(f))
imageio.mimsave('./conf_matrix.gif', images,  duration=0.5)
