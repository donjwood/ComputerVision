from PIL import Image
import numpy as np
import pylab as plt
from common import imtools
from common import harris

# This is the Harris point matching example in Figure 2-2.

im1 = np.array(Image.open("data/crans_1_small.jpg").convert("L"))
im2 = np.array(Image.open("data/crans_2_small.jpg").convert("L"))

# resize to make matching faster
im1 = imtools.imresize(im1,(int(im1.shape[1]/2),int(im1.shape[0]/2)))
im2 = imtools.imresize(im2,(int(im2.shape[1]/2),int(im2.shape[0]/2)))

wid = 5
harrisim = harris.compute_harris_response(im1,5) 
filtered_coords1 = harris.get_harris_points(harrisim,wid+1) 
d1 = harris.get_descriptors(im1,filtered_coords1,wid)

harrisim = harris.compute_harris_response(im2,5) 
filtered_coords2 = harris.get_harris_points(harrisim,wid+1) 
d2 = harris.get_descriptors(im2,filtered_coords2,wid)

print('starting matching')
matches = harris.match_twosided(d1,d2)

plt.figure()
plt.gray() 
harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches) 
plt.show()