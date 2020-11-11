from PIL import Image
import numpy as np
import pylab as pl
from scipy.ndimage import filters, measurements, morphology
from common import imtools

im = np.array(Image.open('data/houses.png').convert('L'))
im = 1*(im<128)

labels, nbr_objects = measurements.label(im)
print('Number of objects:', nbr_objects)

im_open = morphology.binary_opening(im, np.ones((9,5)),iterations=2)
labels_open, nbr_objects_open = measurements.label(im_open)
print("Number of objects:", nbr_objects_open)

pl.figure('Labels')
pl.gray()
pl.title('Labels')
pl.imshow(labels)

pl.figure('Opened Image Labels')
pl.gray()
pl.title('Opened Image Labels')
pl.imshow(labels_open)

pl.show()