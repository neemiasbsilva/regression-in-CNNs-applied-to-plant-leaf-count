import numpy as np
import matplotlib.pyplot as plt

width = 200
height = 200
sigma = 7.

positions = np.array(
	[[10, 10],
	[50, 100],
	[150, 30]]
	)

def get_heamap(positions, width, height, sigma):
	grid_x = np.arange(width)
	grid_y = np.arange(height)
	double_sigma2 = 2 * sigma * sigma
	heatmap = np.zeros((width, height), dtype=np.float)
	for i in range(positions.shape[0]):
		exp_x = np.exp(-(grid_x-positions[i,0])**2/double_sigma2)
		exp_y = np.exp(-(grid_y-positions[i,1])**2/double_sigma2)
		exp = np.outer(exp_y, exp_x)
		heatmap = np.maximum(heatmap, exp)
	plt.imshow(heatmap, cmap='hot', interpolation='nearest')
	plt.savefig('output2.png')
	plt.show()
	return heatmap

heatmap = get_heamap(positions, width, height, sigma)

from keras.applications.inception_v3 import InceptionV3
from keras.layers import UpSampling2D, Conv2D

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = UpSampling2D()(x)
x = Conv2D()(x)
x = Conv2D()(x)
x = UpSampling2D()(x)
x = Conv2D()(x)
x = Conv2D(filters=1,kernel_size=(1,1,1),activation='sigmoid')(x)

model = Model(input=base_model.input, output=x)

X, Y
