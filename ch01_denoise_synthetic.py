from numpy import *
from numpy import random
from scipy.ndimage import filters
import pylab

import imtools

# create synthetic image with noise
im = zeros((500,500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255
im = im + 30*random.standard_normal((500,500))

U,T = imtools.denoise(im,im)
G = filters.gaussian_filter(im,10)

# output the result
pylab.figure('Orignal')
pylab.gray()
pylab.imshow(im)
pylab.figure('De-Noised')
pylab.gray()
pylab.imshow(U)
pylab.figure('Gaussian Blur')
pylab.gray()
pylab.imshow(G)
pylab.show()