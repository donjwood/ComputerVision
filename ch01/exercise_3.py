from PIL import Image
import numpy as np
import pylab as pl
from scipy.ndimage import filters
from common import imtools

# An alternative image normalization to histogram equalization is a quotient image.
# A quotient image is obtained by dividing the image with a blurred version
# I/(I * G). Implement this and try it on some sample images.

def quotient_image(im, sigma):
    im_gaus = imtools.gaussian_blur(im, sigma)
    im_quot = im/im_gaus
    im_quot = np.interp(im_quot, [im_quot.min(), im_quot.max()], [0, 255]).astype('uint8')
    return im_quot

sigma = 10

# Load grayscale image
im_gray = np.array(Image.open('data/Einstein.jpg').convert('L'))
im_gray_quotient = quotient_image(im_gray, sigma)

pl.figure('Quotient Image')
pl.gray()
pl.subplot(2, 2, 1)
pl.title('Original Grayscale Image')
pl.imshow(im_gray)
pl.gray()
pl.subplot(2, 2, 2)
pl.title('Grayscale Image Quotient')
pl.imshow(im_gray_quotient)

# Load color image
im_colour = np.array(Image.open('data/Lenna.png'))
im_colour_quotient = quotient_image(im_colour, sigma)

pl.subplot(2, 2, 3)
pl.title('Original Color Image')
pl.imshow(im_colour)
pl.gray()
pl.subplot(2, 2, 4)
pl.title('Color Image Quotient')
pl.imshow(im_colour_quotient)

pl.show()