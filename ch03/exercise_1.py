"""
Create a function that takes the image coordinates of a square (or rectangular)
object, for example a book, a poster or a 2D bar code, and estimates the transform
that takes the rectangle to a full on frontal view in a normalized coordinate
system. Use ginput() or the strongest Harris corners to find the points.
"""

from PIL import Image
from scipy import ndimage
import numpy as np
import pylab as plt

from common import warp
from common import homography

im1 = np.array(Image.open('data/book_perspective.JPG'))
#im1 = np.array(Image.open('data/SmokeyInBox.jpg'))

plt.figure('Warp Rectangle to Front')
plt.title('Please click the corners of the object.')
plt.imshow(im1)

corner_input = plt.ginput(4)
print('You clicked: ' + str(corner_input))

# .T transposes so the input tuples are the array of x coords and array of y coords, flip swaps it so that it is row, col form.
rowcol_corners = np.flip(np.array(corner_input).astype(int).T, 0)

im1_t = warp.rectangle_to_front(im1, rowcol_corners)

plt.title('Transform')
plt.imshow(im1_t)
plt.show()