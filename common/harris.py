from scipy.ndimage import filters
import numpy as np

def compute_harris_response(im, sigma=3):
    # Compute the Harris corner detector response function
    # for each pixel in a graylevel image.

    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0,1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0,1), imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr

