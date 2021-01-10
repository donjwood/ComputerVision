"""
Module for warping images.
"""
import matplotlib.tri as mpltri
import numpy as np
import pylab as plt
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
        true_cols = np.where(row)[0]
        if len(true_cols) > 0:
            row[min(true_cols):max(true_cols)] = True

    # np.where uses the mask to return im1_t where true, im2 where false.
    return np.where(alpha, im1_t, im2)

def alpha_for_triangle(points,m,n):
    """ Creates alpha map of size (m,n)
    for a triangle with corners defined by points
    (given in normalized homogeneous coordinates). """

    alpha = np.zeros((m,n))
    for i in range(min(points[0]),max(points[0])):
        for j in range(min(points[1]),max(points[1])):
            x = np.linalg.solve(points,[i,j,1])
            if min(x) > 0: #all coefficients positive
                alpha[i,j] = 1

    return alpha

def triangulate_points(x,y):
    """ 
    Delaunay triangulation of 2D points. 
    Modified to use matplotlib triangulation function.
    """

    tri = mpltri.Triangulation(x,y)

    #return just the triangles... triangulation object contains point lists that may not fit with how we are mapping points.
    return tri.triangles

def pw_affine(fromim,toim,fp,tp,tri):
    """ Warp triangular patches from an image.
    fromim = image to warp
    toim = destination image
    fp = from points in hom. coordinates
    tp = to points in hom. coordinates
    tri = triangulation. """
    
    im = toim.copy()
    
    # check if image is grayscale or color
    is_color = len(fromim.shape) == 3
    
    # create image to warp to (needed if iterate colors)
    im_t = np.zeros(im.shape, 'uint8')
    
    for t in tri:
        # compute affine transformation
        H = homography.Haffine_from_points(tp[:,t],fp[:,t])
    
        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:,:,col] = ndimage.affine_transform(
                    fromim[:,:,col],H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
        else:
            im_t = ndimage.affine_transform(
                fromim,H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
    
        # alpha for triangle
        alpha = alpha_for_triangle(tp[:,t],im.shape[0],im.shape[1])
    
        # add triangle to image
        im[alpha>0] = im_t[alpha>0]
    
    return im

def plot_mesh(x,y,tri):
    """ Plot triangles. """
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]] # add first point to end
        plt.plot(x[t_ext],y[t_ext],'r')