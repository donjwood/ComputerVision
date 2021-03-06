"""
Warping an image using the scipy ndimage.affine_transform function.
"""

from PIL import Image
from scipy import ndimage
import numpy as np
import pylab as plt

im = np.array(Image.open('data/empire.jpg').convert('L'))
H = np.array([[1.4,0.05,-100], [0.05,1.5,-100], [0,0,1]])
im2 = ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]))

plt.figure("Affine Transform with ndimage")
plt.gray()
plt.imshow(im2)
plt.show()
