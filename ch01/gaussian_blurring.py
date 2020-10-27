from PIL import Image
import numpy as np
from scipy.ndimage import filters
import pylab as pl

im = pl.array(Image.open('images/empire.jpg'))
im2 = pl.zeros(im.shape)
for i in range(3):
    im2[:,:,i] = filters.gaussian_filter(im[:,:,i],5)
im2 = np.uint8(im2)

pl.figure('Gaussian Blurred')
pl.imshow(im2)
pl.show()
