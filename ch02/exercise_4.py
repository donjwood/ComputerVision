"""
Exercise 4

Create copies of an image with different resolutions (for example by halving the
size a few times). Extract SIFT features for each image. Plot and match features
to get a feel for how and when the scale independence breaks down.
"""

from PIL import Image
import numpy as np
import pylab as plt
from common import imtools
from common import sift

# Seems to break down around 8, clearly breaks down at 16.
div_factor = 8

im = np.array(Image.open('data/empire.jpg').convert("L"))
im_resized = imtools.imresize(im, (int(im.shape[1]/div_factor),int(im.shape[0]/div_factor)))

kp1, desc1 = sift.detect_and_compute(im)
kp2, desc2 = sift.detect_and_compute(im_resized)

print('starting matching')
matches = sift.match_twosided(desc1,desc2)

plt.figure()
plt.gray()
sift.plot_matches(im,im_resized,kp1,kp2,matches)
plt.show()
