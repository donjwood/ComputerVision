from PIL import Image
import numpy as np
import pylab as pl
from scipy.ndimage import filters, measurements, morphology
from common import imtools

# Apply the label() function to a thresholded image of your choice. Use histograms
# and the resulting label image to plot the distribution of object sizes in the image.

im = np.array(Image.open('images/aircraft-formation.jpg').convert('L'))
im_bin = 1*(im<128)

#labels, nbr_objects = measurements.label(im_bin)
im_open = morphology.binary_opening(im_bin, np.ones((9,5)),iterations=1)
labels_open, nbr_objects_open = measurements.label(im_open)
print('Number of objects:', nbr_objects_open)

pl.figure('Labels')
pl.subplot(1, 2, 1)
pl.gray()
pl.title('Labeled Image')
pl.imshow(labels_open)
pl.subplot(1, 2, 2)
pl.title('Histogram')
pl.hist(labels_open.flatten(), bins=nbr_objects_open, log=True)


pl.show()