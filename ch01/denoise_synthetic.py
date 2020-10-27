import numpy as np
from numpy import random
from scipy.ndimage import filters
import pylab as pl
from common import imtools

# create synthetic image with noise
im = pl.zeros((500,500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255
im = im + 30*random.standard_normal((500,500))

U,T = imtools.denoise(im,im)
G = filters.gaussian_filter(im,10)

# output the result
pl.figure('Orignal')
pl.gray()
pl.imshow(im)
pl.figure('De-Noised')
pl.gray()
pl.imshow(U)
pl.figure('Gaussian Blur')
pl.gray()
pl.imshow(G)
pl.show()