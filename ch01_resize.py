from PIL import Image
from pylab import *
from imtools import *

im = array(Image.open('empire.jpg'))
figure()
imshow(im)
im_resz = imresize(im, [142,200])
figure()
imshow(im_resz)
show()