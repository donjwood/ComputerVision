from PIL import Image
from numpy import *
from scipy.ndimage import filters
from pylab import *

im = array(Image.open('empire.jpg').convert('L'))

# Sobel derivative filters
imx = zeros(im.shape)
filters.sobel(im, 1, imx)

imy = zeros(im.shape)
filters.sobel(im, 0, imy)

magnitude = sqrt(imx**2 + imy**2)

print ("Sobel Magnitude: " + str(magnitude))

# Gaussian derivative filters
sigma = 5 # Standard Deviation

imgx = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), imgx)
imgy = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), imgy)

figure('Orginal')
gray()
imshow(im)
figure('Sobel X Derivative')
gray()
imshow(imx)
figure('Sobel Y Derivative')
gray()
imshow(imy)
figure('Gaussian X Derivative')
gray()
imshow(imgx)
figure('Gaussian Y Derivative')
gray()
imshow(imgy)
show()