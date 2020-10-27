from PIL import Image
import numpy as np
from scipy.ndimage import filters
import pylab as pl

im = pl.array(Image.open('images/empire.jpg').convert('L'))

# Sobel derivative filters
imx = pl.zeros(im.shape)
filters.sobel(im, 1, imx)

imy = pl.zeros(im.shape)
filters.sobel(im, 0, imy)

magnitude = np.sqrt(imx**2 + imy**2)

print ("Sobel Magnitude: " + str(magnitude))

# Gaussian derivative filters
sigma = 5 # Standard Deviation

imgx = pl.zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), imgx)
imgy = pl.zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), imgy)

pl.figure('Orginal')
pl.gray()
pl.imshow(im)
pl.figure('Sobel X Derivative')
pl.gray()
pl.imshow(imx)
pl.figure('Sobel Y Derivative')
pl.gray()
pl.imshow(imy)
pl.figure('Gaussian X Derivative')
pl.gray()
pl.imshow(imgx)
pl.figure('Gaussian Y Derivative')
pl.gray()
pl.imshow(imgy)
pl.show()