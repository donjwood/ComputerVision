"""
Create a panorama with the university photos.
"""

import numpy as np
import pylab as plt 

from common import homography
from common import sift
from common import warp
from PIL import Image

imname = ['data/Univ'+str(i+1)+'.jpg' for i in range(5)]

im = {}
kp = {}
desc = {}

for i in range(5):
    im[i] = np.array(Image.open(imname[i]))
    kp[i], desc[i] = sift.detect_and_compute(im[i])

matches = {}
for i in range(4):
    matches[i] = sift.match(desc[i+1],desc[i])

# function to convert the matches to hom. points
def convert_points(j):
    ndx = matches[j].nonzero()[0]
    fp = homography.make_homog(kp[j+1][ndx,:2].T)
    ndx2 = [int(matches[j][i]) for i in ndx]
    tp = homography.make_homog(kp[j][ndx2,:2].T)
    return fp,tp

# estimate the homographies
model = homography.RansacModel()
fp,tp = convert_points(1)
H_12 = homography.H_from_ransac(fp,tp,model)[0] #im 1 to 2
fp,tp = convert_points(0)
H_01 = homography.H_from_ransac(fp,tp,model)[0] #im 0 to 1
tp,fp = convert_points(2) #NB: reverse order
H_32 = homography.H_from_ransac(fp,tp,model)[0] #im 3 to 2
tp,fp = convert_points(3) #NB: reverse order
H_43 = homography.H_from_ransac(fp,tp,model)[0] #im 4 to 3

delta = 2000 #for padding and translation
im1 = np.array(Image.open(imname[1]))
im2 = np.array(Image.open(imname[2]))
im_12 = warp.panorama(H_12,im1,im2,delta,delta)
im1 = np.array(Image.open(imname[0]))
im_02 = warp.panorama(np.dot(H_12,H_01),im1,im_12,delta,delta)
im1 = np.array(Image.open(imname[3]))
im_32 = warp.panorama(H_32,im1,im_02,delta,delta)
im1 = np.array(Image.open(imname[4]))
im_42 = warp.panorama(np.dot(H_32,H_43),im1,im_32,delta,2*delta)

plt.figure("University Panorama")
plt.imshow(im_42.astype('uint8'))
plt.axis('off')
plt.show()
