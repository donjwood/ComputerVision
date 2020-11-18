from PIL import Image
import numpy as np
import pylab as plt
from common import imtools
from common import sift

im1 = np.array(Image.open("data/climbing_1_small.jpg").convert("L"))
im2 = np.array(Image.open("data/climbing_2_small.jpg").convert("L"))

kp1, desc1 = sift.detect_and_compute(im1)
kp2, desc2 = sift.detect_and_compute(im2)

print('starting matching')
matches = sift.match_twosided(desc1,desc2)

plt.figure("SIFT Rock Wall 2-Side Matching")
plt.gray() 
sift.plot_matches(im1,im2,kp1,kp2,matches) 
plt.show()