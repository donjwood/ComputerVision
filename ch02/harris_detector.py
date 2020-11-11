from PIL import Image
import numpy as np
import pylab as plt
from common import harris

threshold = 0.1

im = np.array(Image.open('data/empire.jpg').convert('L'))
harrisim = harris.compute_harris_response(im)
filtered_coords = harris.get_harris_points(harrisim,6,threshold)
harris.plot_harris_points(im, filtered_coords)

# plt.figure("Harris Corner Detector")
# plt.gray()
# #plt.subplot(2, 2, 1)
# plt.title('Harris Response')
# plt.imshow(harrisim)

# plt.show()