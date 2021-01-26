"""
Sift Matching on the university photos.
"""

import numpy as np
import pylab as plt 

from common import sift
from PIL import Image
imname = ['data/Univ'+str(i+1)+'.jpg' for i in range(5)]

im = {}
kp = {}
desc = {}

for i in range(5):
    im[i] = np.array(Image.open(imname[i]))
    kp[i], desc[i] = sift.detect_and_compute(im[i])

print('starting matching')
plt.figure("University Image SIFT Matches")

matches = {}
for i in range(4):
    plt.subplot(2,2,i+1)
    matches[i] = sift.match(desc[i+1],desc[i])
    sift.plot_matches(im[i+1],im[i],kp[i+1],kp[i],matches[i]) 

plt.show()