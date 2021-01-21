"""
Perform Principal Component Analysis on font images.
"""
import numpy as np
import os
import pylab as plt

from PIL import Image
from common import pca

font_images_dir = 'data/selectedfontimages'

imlist = os.listdir(font_images_dir)

im = np.array(Image.open(font_images_dir + '/' + imlist[0])) # open one image to get size
m,n = im.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images
# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(font_images_dir + '/' + im)).flatten()
            for im in imlist],'f')

# perform PCA
V,S,immean = pca.pca(immatrix)

# show some images (mean and 7 first modes)
plt.figure()
plt.gray()
plt.subplot(2,4,1)
plt.imshow(immean.reshape(m,n))
for i in range(7):
    plt.subplot(2,4,i+2)
    plt.imshow(V[i].reshape(m,n))

plt.show()