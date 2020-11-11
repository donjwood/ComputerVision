from PIL import Image
import numpy as np
import pylab as pl
from scipy.ndimage import filters
from common import imtools

# Write a function that finds the outline of simple objects in images (for example a
# square against white background) using image gradients.

# Stumbled on how to compute the norm of an image here (https://www.sciencedirect.com/topics/computer-science/sobel-filter)
# https://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy

def outlines(im):
    x = filters.sobel(im.astype('int32'), axis=1)
    y = filters.sobel(im.astype('int32'), axis=0)
    im_out = np.sqrt(np.square(x) + np.square(y))
    im_out *= 255.0 / np.max(im_out)
    return im_out


im = np.zeros((500, 500), dtype=np.uint8)
im.fill(255)
im[100:400,100:400] = 128
im[200:300,200:300] = 255

pl.figure('Outline')
pl.subplot(2, 2, 1)
pl.gray()
pl.title('Square')
pl.axis('off')
pl.imshow(im, vmin=0, vmax=255)

pl.subplot(2, 2, 2)
pl.gray()
pl.title('Square Outlines')
pl.axis('off')
pl.imshow(outlines(im), vmin=0, vmax=255)

im2 = np.array(Image.open('data/SmokeyInBox.jpg').convert('L'))
pl.subplot(2, 2, 3)
pl.gray()
pl.title('Image')
pl.axis('off')
pl.imshow(im2, vmin=0, vmax=255)

pl.subplot(2, 2, 4)
pl.gray()
pl.title('Image Outlines')
pl.axis('off')
pl.imshow(outlines(im2), vmin=0, vmax=255)

pl.show()