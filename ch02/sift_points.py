from PIL import Image
import numpy as np
import pylab as plt
from common import imtools
from common import sift

im = np.array(Image.open('data/empire.jpg'))
kp, desc = sift.detect_and_compute(im)
plt.figure("SIFT Key Points")
sift.plot_features(im,kp, True)
plt.show()