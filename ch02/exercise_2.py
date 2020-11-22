"""
Exercise 2

Incrementally apply stronger blur (or ROF de-noising) to an image and extract
Harris corners. What happens?
"""
from PIL import Image
import numpy as np
import pylab as plt
from common import harris
from common import imtools

threshold = 0.1
sigma = 5
loops = 2

im = np.array(Image.open('data/empire.jpg').convert('L'))
blur_im = im
for x in range(loops):
    blur_im,T = imtools.denoise(blur_im, blur_im)
    #blur_im = imtools.gaussian_blur(blur_im, sigma)

harrisim = harris.compute_harris_response(blur_im)
filtered_coords = harris.get_harris_points(harrisim,6,threshold)
harris.plot_harris_points(blur_im, filtered_coords)
