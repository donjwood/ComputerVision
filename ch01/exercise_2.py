from PIL import Image
import numpy as np
import pylab as pl
from scipy.ndimage import filters
from common import imtools

# Unsharp
def unsharp(im, sigma, amount, threshold):

    im_gaus = imtools.gaussian_blur(im, sigma)
    mask = (im - im_gaus) * amount
    mask = (abs(mask) > threshold) * mask
    return np.clip(im + mask, 0, 255).astype('uint8')


sigma = 5
amount= 1.5
threshold = 10

# Load original image
im_gray = np.array(Image.open('images/SmokeyInBox.jpg').convert('L')).astype('float32')
im_gray_unsharp = unsharp(im_gray, sigma, amount, threshold)

pl.figure('Original Grayscale Image')
pl.gray()
pl.title('Original Grayscale Image')
pl.imshow(im_gray)

pl.figure('Unsharped Grayscale')
pl.gray()
pl.title('Unsharped Grayscale Image')
pl.imshow(im_gray_unsharp)

# Load original color image
im_color = np.array(Image.open('images/SmokeyInBox.jpg')).astype('float32')

im_color_unsharp = unsharp(im_color, sigma, amount, threshold)

pl.figure('Original Color Image')
pl.gray()
pl.title('Original Color Image')
pl.imshow(im_color.astype('uint8'))

pl.figure('Unsharped Color Image')
pl.gray()
pl.title('Unsharped Color Image')
pl.imshow(im_color_unsharp)


pl.show()