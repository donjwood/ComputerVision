from PIL import Image
import numpy as np
import pylab as pl
from scipy.ndimage import filters

# Define range of sigma values
sigma_range = range(5, 25, 5)

# Load original image
im = np.array(Image.open('images/SmokeyInBox.jpg').convert('L'))

# Output Original with contour
pl.figure('Original')
pl.gray()
pl.title('Original Image')
pl.imshow(im)
pl.figure('Original Image Contour')
pl.title('Original Image Contour')
pl.contour(im, origin='image')

# Output blurred with different sigma values
for sigma in sigma_range:
    im_gaus = np.zeros(im.shape)
    im_gaus[:,:] = filters.gaussian_filter(im[:,:],sigma)
    im_gaus = np.uint8(im_gaus)
    pl.figure('Gaussian Blur with \u03C3 = ' + str(sigma))
    pl.gray()
    pl.title('Gaussian Blur with \u03C3 = ' + str(sigma))
    pl.imshow(im_gaus)
    pl.figure('Contour of Image with \u03C3 = ' + str(sigma))
    pl.title('Contour of Image with \u03C3 = ' + str(sigma))
    pl.contour(im_gaus, origin='image')

pl.show()