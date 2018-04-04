import numpy as np
import matplotlib.pyplot as plt


def N(x, mu, s):
	return np.exp(-(x-mu)**2/s**2)

x = np.linspace(-10, 10, 1000)

ax = plt.subplot(111)

ax.plot(x, N(x, 0, 4.8)*np.sin(x), c='r')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.annotate('Global Minima', xy=(-1.446, -0.906), xytext=(-6, -2),
            arrowprops=dict(facecolor='blue', width=0.1, headlength=0.1,headwidth=0.1),
            )
ax.annotate('Local Minima', xy=(4.351, -0.411), xytext=(5, 1),
            arrowprops=dict(facecolor='blue', width=0.1, headlength=0.1,headwidth=0.1),
            )
plt.xticks([],[])
plt.yticks([],[])
ax.set_ylim(-2,2)
plt.xlabel('J')
plt.ylabel('\\theta')
plt.show()
