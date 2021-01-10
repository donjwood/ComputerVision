"""
Example of piecewise affine warping using Delaunay triangulated points.
"""

from PIL import Image
from scipy import ndimage
import numpy as np
import pylab as plt
import matplotlib.tri as mpltri

from common import warp
from common import homography

# open image to warp
fromim = np.array(Image.open('data/sunset_tree.jpg'))
x,y = np.meshgrid(range(5),range(6))
x = (fromim.shape[1]/4) * x.flatten()
y = (fromim.shape[0]/5) * y.flatten()

# triangulate
tri = warp.triangulate_points(x,y)

# open image and destination points
im = np.array(Image.open('data/turningtorso1.jpg'))
tp = np.loadtxt('data/turningtorso1_points.txt', dtype=int) # destination points

# convert points to hom. coordinates
fp = np.vstack((y,x,np.ones((1,len(x)))))
tp = np.vstack((tp[:,1],tp[:,0],np.ones((1,len(tp)), dtype=int)))

# warp triangles
im = warp.pw_affine(fromim,im,fp,tp,tri)

# plot
plt.figure()
plt.imshow(im)
warp.plot_mesh(tp[1],tp[0],tri)
#plt.triplot(tri)
plt.axis('off')
plt.show()