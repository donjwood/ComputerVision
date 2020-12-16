"""
Module for warping images.
"""
import numpy as np
from scipy import ndimage
from common import homography

def image_in_image(im1,im2,tp):
    """ Put im1 in im2 with an affine transformation
    such that corners are as close to tp as possible.
    tp are homogeneous and counter-clockwise from top left. """

    # points to warp from
    m,n = im1.shape[:2]
    fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

    # compute affine transform and apply
    H = homography.Haffine_from_points(tp,fp)
    im1_t = ndimage.affine_transform(im1,H[:2,:2],
            (H[0,2],H[1,2]),im2.shape[:2])
    alpha = (im1_t > 0)

    # Fix the alpha mask which is false anywhere the embedded image is black.
    # This will make any columns true between the first and last true in a row.
    for row in alpha:
        first_true_col = 0
        last_true_col = 0
        for idx, col in enumerate(row):
            if col:
                if first_true_col == 0:
                    first_true_col = idx
                last_true_col = idx

        if last_true_col !=0:
            row[first_true_col:last_true_col] = True

    return (1-alpha)*im2 + alpha*im1_t
