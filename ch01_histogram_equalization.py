from PIL import Image
from numpy import *
import imtools
from pylab import *

im = array(Image.open('empire.jpg').convert('L'))
im2,cdf = imtools.histeq(im)
figure('Original')
gray()
imshow(im)
figure('Original Histogram')
hist(im.flatten(), 128)
figure('Equalized Image')
gray()
imshow(im2)
figure('Equalized Histogram')
hist(im2.flatten(), 128)
show()