from PIL import Image
import pylab as pl
import numpy as np
from common import imtools 

im = np.array(Image.open('data/empire.jpg'))
pl.figure('Orignal')
pl.imshow(im)
im_resz = imtools.imresize(im, [142,200])
pl.figure('Resized')
pl.imshow(im_resz)
pl.show()