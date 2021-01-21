"""
Show mean of faces images.
"""
import numpy as np
import os
import pylab as plt 

from PIL import Image
from common import pca

def get_mean_image(im_dir):

    # Get all images in directory (skip subdirectory)
    imlist = [f for f in os.listdir(im_dir) if os.path.isfile(os.path.join(im_dir, f))]

    # Open one image to get size
    im = np.array(Image.open(im_dir + '/' + imlist[0]))
    m,n = im.shape[0:2]

    # Flatten for PCA
    immatrix = np.array([np.array(Image.open(im_dir + '/' + imlist[i]).convert('L')).flatten()
    for i in range(150)],'f')

    # perform PCA
    V,S,immean = pca.pca(immatrix)

    return immean.reshape(m,n)

faces_dir = 'data/jkfaces'
aligned_faces_dir = 'data/jkfaces/aligned'

unaligned_mean = get_mean_image(faces_dir)
aligned_mean = get_mean_image(aligned_faces_dir)

# show some images (mean and 7 first modes)
plt.figure("Mean of Unaligned and Aligned Face Images")
plt.gray()

plt.subplot(1,2,1)
plt.title("Unaligned")
plt.imshow(unaligned_mean)

plt.subplot(1,2,2)
plt.title("Aligned")
plt.imshow(aligned_mean)


plt.show()