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

im1 = np.array(Image.open('data/book_perspective.JPG').convert('L'))

plt.title('Please click the corners of the object in the image counter-clockwise from the top left.')
plt.imshow(im1)

corner_input = plt.ginput(4)
print('You clicked: ' + str(corner_input))

# .T transposes so the input tuples are the array of x coords and array of y coords, flip swaps it so that it is row, col form.
fp = np.flip(np.array(corner_input).astype(int).T, 0)

# Find max row, col values
max_fp = np.amax(fp, 1)

# Find min row, col values
min_fp = np.amin(fp, 1)

to_height, to_width = max_fp - min_fp
#print(fp)
#print(max_fp)
#print(min_fp)
#print(str(to_height) + ' ' + str(to_width))

fp = np.vstack((fp, np.ones(4)))
tp = np.array([[0,to_height,to_height,0],[0,0,to_width,to_width],[1,1,1,1]])

#print(fp)
#print(tp)

H = homography.Haffine_from_points(tp,fp)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
                                 (H[0,2],H[1,2]),(to_height,to_width))

plt.title('Transform')
plt.imshow(im1_t)
plt.show()