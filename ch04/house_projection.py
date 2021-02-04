"""
Projection of points from house.p3d example.
"""
import numpy as np
import pylab as plt

from common import camera

# load points
points = np.loadtxt('data/house.p3d').T
points = np.vstack((points,np.ones(points.shape[1])))

# setup camera
P = np.hstack((np.eye(3),np.array([[0],[0],[-10]])))
cam = camera.Camera(P)
x = cam.project(points)

# plot projection
plt.figure('House Projection')
plt.subplot(1, 2, 1)
plt.title('Projected Points')
plt.plot(x[0],x[1],'k.')

# create transformation
r = 0.05*np.random.rand(3)
rot = camera.rotation_matrix(r)

# rotate camera and project
plt.subplot(1,2,2)
plt.title('Camera Rotation')
for t in range(20):
    cam.P = np.dot(cam.P,rot)
    x = cam.project(points)
    plt.plot(x[0],x[1],'k.')

plt.show()