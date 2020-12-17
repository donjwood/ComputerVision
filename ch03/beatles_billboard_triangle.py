"""
Image warp example of putting Beatles image on a billboard at angle requiring .
"""

from PIL import Image
from scipy import ndimage
import numpy as np
import pylab as plt

from common import warp
from common import homography

# example of affine warp of im1 onto im2
im1 = np.array(Image.open('data/beatles.jpg').convert('L'))
im2 = np.array(Image.open('data/billboard_for_rent_2.jpg').convert('L'))

# set to points
tp = np.array([[313,572,462,131],[306,295,1243,1246],[1,1,1,1]])

# set from points to corners of im1
m,n = im1.shape[:2]
fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

# first triangle
tp2 = tp[:,:3]
fp2 = fp[:,:3]

# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
                                 (H[0,2],H[1,2]),im2.shape[:2])

# alpha for triangle
alpha = warp.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im3 = (1-alpha)*im2 + alpha*im1_t

# second triangle
tp2 = tp[:,[0,2,3]]
fp2 = fp[:,[0,2,3]]
# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
                                 (H[0,2],H[1,2]),im2.shape[:2])

# alpha for triangle
alpha = warp.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im4 = (1-alpha)*im3 + alpha*im1_t

plt.figure('Beatles Billboard')
plt.gray()
plt.imshow(im4)
plt.axis('equal')
plt.axis('off')
plt.show()
