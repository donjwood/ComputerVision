from PIL import Image
from pylab import *
import imtools

im = array(Image.open('empire.jpg'))
figure('Orignal')
imshow(im)
im_resz = imtools.imresize(im, [142,200])
figure('Resized')
imshow(im_resz)
show()