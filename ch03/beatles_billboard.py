"""
Image warp example of putting Beatles image on a billboard.
"""

from PIL import Image
from scipy import ndimage
import numpy as np
import pylab as plt

from common import warp

# example of affine warp of im1 onto im2
im1 = np.array(Image.open('data/beatles.jpg').convert('L'))
im2 = np.array(Image.open('data/billboard_for_rent.jpg').convert('L'))

# set to points
tp = np.array([[18,260,256,16],[103,90,613,592],[1,1,1,1]])

im3 = warp.image_in_image(im1,im2,tp)

plt.figure('Beatles Billboard')
plt.gray()
plt.imshow(im3)
plt.axis('equal')
plt.axis('off')
plt.show()
