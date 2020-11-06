from PIL import Image
import numpy as np
import pylab as plt
from common import harris

im = np.array(Image.open('images/empire.jpg').convert('L'))
harrisim = harris.compute_harris_response(im)

plt.figure("Harris Corner Detector")
plt.gray()
#plt.subplot(2, 2, 1)
plt.title('Harris Response')
plt.imshow(harrisim)

plt.show()