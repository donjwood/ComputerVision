from PIL import Image
import pylab as pl
from common import imtools 

im = pl.array(Image.open('images/empire.jpg'))
pl.figure('Orignal')
pl.imshow(im)
im_resz = imtools.imresize(im, [142,200])
pl.figure('Resized')
pl.imshow(im_resz)
pl.show()