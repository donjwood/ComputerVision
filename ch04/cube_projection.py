"""
Example of projecting a 3D cube onto a book cover.
"""
from PIL import Image
from scipy import ndimage
import numpy as np
import pylab as plt

from common import objects3d
from common import homography
from common import camera
from common import sift

im_book_frontal = np.array(Image.open('data/book_frontal.JPG'))
im_book_perspective = np.array(Image.open('data/book_perspective.JPG'))

# Compute features
l0, d0 = sift.detect_and_compute(im_book_frontal)
l1, d1 = sift.detect_and_compute(im_book_perspective)

# Match features and estimate homography
matches = sift.match_twosided(d0,d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx,:2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2,:2].T)

model = homography.RansacModel()
H = homography.H_from_ransac(fp, tp, model)[0]
# camera calibration
K = camera.book_calibration((747,1000))

# 3D points at plane z=0 with sides of length 0.2
box = objects3d.cube_points([0,0,0.1],0.1)

# project bottom square in first image
cam1 = camera.Camera(np.hstack((K,np.dot(K,np.array([[0],[0],[-1]])))))
# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:,:5]))

# use H to transfer points to the second image
box_trans = homography.normalize(np.dot(H,box_cam1))

# compute second camera matrix from cam1 and H
cam2 = camera.Camera(np.dot(H,cam1.P))
A = np.dot(np.linalg.inv(K),cam2.P[:,:3])
A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = np.dot(K,A)

# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))

# test: projecting point on z=0 should give the same
point = np.array([1,1,0,1]).T
print(homography.normalize(np.dot(np.dot(H,cam1.P),point)))
print(cam2.project(point))

# 2D projection of bottom square
plt.figure()
plt.imshow(im_book_frontal)
plt.plot(box_cam1[0,:],box_cam1[1,:],linewidth=3)

# 2D projection transferred with H
plt.figure()
plt.imshow(im_book_perspective)
plt.plot(box_trans[0,:],box_trans[1,:],linewidth=3)

# 3D cube
plt.figure()
plt.imshow(im_book_perspective)
plt.plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
plt.show()
