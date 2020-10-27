from PIL import Image
import numpy as np
from common import imtools
import pylab as pl

im = pl.array(Image.open('images/empire.jpg').convert('L'))
im2,cdf = imtools.histeq(im)
pl.figure('Original')
pl.gray()
pl.imshow(im)
pl.figure('Original Histogram')
pl.hist(im.flatten(), 128)
pl.figure('Equalized Image')
pl.gray()
pl.imshow(im2)
pl.figure('Equalized Histogram')
pl.hist(im2.flatten(), 128)
pl.show()