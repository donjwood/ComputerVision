from PIL import Image
import numpy as np
import pylab as pl
from scipy.ndimage import filters
from common import imtools

def quotient_image(im, sigma):
    im_gaus = imtools.gaussian_blur(im, sigma)
    im_quot = im/im_gaus
    return np.clip(im_quot, 0, 255).astype('uint8')

sigma = 5

# Load grayscale image
im_gray = np.array(Image.open('images/SmokeyInBox.jpg').convert('L')).astype('float32')
im_gray_quotient = quotient_image(im_gray, sigma)

pl.figure('Original Grayscale Image')
pl.gray()
pl.title('Original Grayscale Image')
pl.imshow(im_gray)
pl.figure('Grayscale Image Quotient')
pl.gray()
pl.title('Grayscale Image Quotient')
pl.imshow(im_gray_quotient)

# Load color image
im = np.array(Image.open('images/SmokeyInBox.jpg')).astype('float32')
im_quotient = quotient_image(im, sigma)

pl.figure('Original Image')
pl.gray()
pl.title('Original Image')
pl.imshow(im.astype('uint8'))
pl.figure('Image Quotient')
pl.gray()
pl.title('Image Quotient')
pl.imshow(im_quotient)


pl.show()