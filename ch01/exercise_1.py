from PIL import Image
import numpy as np
import pylab as pl
from scipy.ndimage import filters
from common import imtools

# Take an image and apply Gaussian blur like in Figure 1.9. Plot the image contours
# for increasing values of sigma. What happens? Can you explain why?

# Define range of sigma values
sigma_range = range(5, 25, 5)

# Load original image
im = np.array(Image.open('data/SmokeyInBox.jpg').convert('L'))

# Output Original with contour
pl.figure('Original Image and Contour')
pl.subplot(1, 2, 1)
pl.gray()
pl.title('Original Image')
pl.imshow(im)
pl.subplot(1, 2, 2)
pl.title('Original Image Contour')
pl.contour(im, origin='image')

# Output blurred with different sigma values
for sigma in sigma_range:
    im_gaus = imtools.gaussian_blur(im, sigma)
    pl.figure('Blurred with \u03C3 = ' + str(sigma))
    pl.subplot(1, 2, 1)
    pl.gray()
    pl.title('Gaussian Blur with \u03C3 = ' + str(sigma))
    pl.imshow(im_gaus)
    pl.subplot(1, 2, 2)
    pl.title('Contour of Image with \u03C3 = ' + str(sigma))
    pl.contour(im_gaus, origin='image')

pl.show()