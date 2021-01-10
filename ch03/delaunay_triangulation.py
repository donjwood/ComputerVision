from PIL import Image
from scipy import ndimage
import numpy as np
import pylab as plt
import matplotlib.tri as md

"""
Delaunay triangulation as described in the book doesn't seem to work anymore.
This instead uses the Triangulation function in the matplotlib.tri library.
https://codeloop.org/python-matplotlib-plotting-triangulation/
https://matplotlib.org/3.1.0/api/tri_api.html
"""

x,y = np.array(np.random.standard_normal((2,100)))
"""
I'm not sure how this "triangles" output works. It does contain an array "triangles" that
contains arrays of 3 integers. My guess is the three integers are the indices of the input
x,y arrays that correspond to the 3 vertices of the triangle.
"""
triangles = md.Triangulation(x,y)

plt.figure()
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.triplot.html
plt.triplot(triangles)
plt.axis('off')
plt.show()
