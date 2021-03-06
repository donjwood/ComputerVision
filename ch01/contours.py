from PIL import Image
import numpy as np
import pylab as pl

# read image to array
im = np.array(Image.open('data/empire.jpg').convert('L'))

# create a new figure
pl.figure('Contours')
# don't use colors
pl.gray()
# show contours with origin upper left corner
pl.contour(im, origin='image')

pl.axis('equal')
pl.axis('off')

pl.figure('Histogram')
pl.hist(im.flatten(), 128)
pl.show()