from PIL import Image
import numpy as np
import pylab as plt
from scipy.ndimage import filters
from common import imtools

# An alternative image normalization to histogram equalization is a quotient image.
# A quotient image is obtained by dividing the image with a blurred version
# I/(I * G). Implement this and try it on some sample images.

sigma = 10

im = np.array(Image.open('data/Einstein.jpg').convert('L'))
quot = im/filters.gaussian_filter(im, sigma)
quot = np.interp(quot, [quot.min(), quot.max()], [0, 255]).astype('uint8')
plt.imshow(quot)
plt.show()
