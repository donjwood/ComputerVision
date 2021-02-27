"""
Example of camera matrix factorization.
"""

import numpy as np

from common import camera

K = np.array([[1000,0,500],[0,1000,300],[0,0,1]])
tmp = camera.rotation_matrix([0,0,1])[:3,:3]
Rt = np.hstack((tmp,np.array([[50],[40],[30]])))
cam = camera.Camera(np.dot(K,Rt))
cam_K, cam_R, cam_t = cam.factor()
print('K:')
print(K)
print('Rt:')
print(Rt)
print('Camera Factor K:')
print(cam_K)
print('Camera Factor R:')
print(cam_R)
print('Camera Factor t:')
print(cam_t)
print('Camera P:')
print(cam.P)