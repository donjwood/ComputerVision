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
    alpha = alpha_for_warped_image(im1_t)

    # Fix the alpha mask which is false anywhere the embedded image is black.
    # This will make any columns true between the first and last true in a row.
    #for row in alpha:
    #    true_cols = np.where(row)[0]
    #    if len(true_cols) > 0:
    #        row[min(true_cols):max(true_cols)] = True

    # np.where uses the mask to return im1_t where true, im2 where false.
    return np.where(alpha, im1_t, im2)

def alpha_for_warped_image(warped_im):
    """
    Solution for chapter 3, exercise 2.
    Takes a warped image and computes the alpha map
    for blending.
    """
    # Creates the initial alpha mask by setting it to true wherever the
    # value in the image is greater than 0. The problem is then the 
    # black points in the image will be 0 in the alpha map.

    alpha = (warped_im > 0)

    # Loop through all the rows in the alpha map and find out if there
    # are any true values. If there are, make all values between the minimum true
    # column and maximum true column true. This will eliminate the false values
    # created by black pixels.
    for row in alpha:
        true_cols = np.where(row)[0]
        if len(true_cols) > 0:
            row[min(true_cols):max(true_cols)] = True

    return alpha

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

def panorama(H,fromim,toim,padding=2400,delta=2400):
    """ Create horizontal panorama by blending two images
    using a homography H (preferably estimated using RANSAC).
    The result is an image with the same height as toim. ’padding’
    specifies number of fill pixels and ’delta’ additional translation. """

    # check if images are grayscale or color
    is_color = len(fromim.shape) == 3
    
    # homography transformation for geometric_transform()
    def transf(p):
        p2 = np.dot(H,[p[0],p[1],1])
        return (p2[0]/p2[2],p2[1]/p2[2])

    if H[1,2]<0: # fromim is to the right
        print('warp - right')
        # transform fromim
        if is_color:
            # pad the destination image with zeros to the right
            toim_t = np.hstack((toim,np.zeros((toim.shape[0],padding,3))))
            fromim_t = np.zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],
                transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the right
            toim_t = np.hstack((toim,np.zeros((toim.shape[0],padding))))
            fromim_t = ndimage.geometric_transform(fromim,transf,
                        (toim.shape[0],toim.shape[1]+padding))
    else:
        print('warp - left')
        # add translation to compensate for padding to the left
        H_delta = np.array([[1,0,0],[0,1,-delta],[0,0,1]])
        H = np.dot(H,H_delta)
        # transform fromim
        if is_color:
            # pad the destination image with zeros to the left
            toim_t = np.hstack((np.zeros((toim.shape[0],padding,3)),toim))
            fromim_t = np.zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],
                    transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the left
            toim_t = np.hstack((np.zeros((toim.shape[0],padding)),toim))
            fromim_t = ndimage.geometric_transform(fromim,
                transf,(toim.shape[0],toim.shape[1]+padding))

    # blend and return (put fromim above toim)
    if is_color:
        # all non black pixels
        alpha = ((fromim_t[:,:,0] * fromim_t[:,:,1] * fromim_t[:,:,2] ) > 0)
        for col in range(3):
            toim_t[:,:,col] = fromim_t[:,:,col]*alpha + toim_t[:,:,col]*(1-alpha)
    else:
        alpha = (fromim_t > 0)
        toim_t = fromim_t*alpha + toim_t*(1-alpha)

    return toim_t
    
def rectangle_to_front (fromim, rowcol_corners):
    """
    Chapter 3, exercise 1 solution.
    Takes an image and row/col corner points of a rectangle 
    from top-left to top-right counter-clockwise and
    translates the image to the front.
    """

    # check if images are grayscale or color
    is_color = len(fromim.shape) == 3

    # Find max row, col values
    max_fp = np.amax(rowcol_corners, 1)

    # Find min row, col values
    min_fp = np.amin(rowcol_corners, 1)

    # Find difference. This will be the dimensions of the new image.
    to_height, to_width = max_fp - min_fp

    # Create matrices of from points and to points
    fp = np.vstack((rowcol_corners, np.ones(4)))
    tp = np.array([[0,to_height,to_height,0],[0,0,to_width,to_width],[1,1,1,1]])

    # Use corners to create triangles. There are only 4 points to work with, so splitting into two 
    # triangles like the Beatles billboard example.
    x = [0,0,to_width,to_width]
    y = [0,to_height,to_height,0]
    tri = triangulate_points(x,y)

    # Create empty image
    if is_color:
        im_empty = np.zeros([to_height,to_width,3],dtype=np.uint8)
    else:
        im_empty = np.zeros([to_height,to_width],dtype=np.uint8)

    # Warp with triangles
    toim = pw_affine(fromim,im_empty,fp,tp,tri)

    return toim