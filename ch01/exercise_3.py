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
    im_quot = im/np.clip(im_gaus, 1, 255)
    #return im_quot
    #return np.interp(im_quot, [im_quot.min(), im_quot.max()], [0, 1])
    return np.clip(im + im_quot, 0, 255).astype('uint8')

sigma = 5

# Load grayscale image
im_gray = np.array(Image.open('data/SmokeyInBox.jpg').convert('L')).astype('float32')
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
im = np.array(Image.open('data/SmokeyInBox.jpg')).astype('float32')
im_quotient = quotient_image(im, sigma)
pl.subplot(2, 2, 3)
pl.title('Original Image')
pl.imshow(im.astype('uint8'))
pl.subplot(2, 2, 4)
pl.title('Image Quotient')
pl.imshow(im_quotient)


pl.show()