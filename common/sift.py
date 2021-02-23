from PIL import Image
import cv2 as cv
import numpy as np
import pylab as plt
import os


def detect_and_compute(im, edge_threshold = 10, contrast_threshold=0.04):
    """ Detect and compute the keypoints and descriptors of an image using SIFT. """

    #Create OpenCV SIFT with specified parameters.
    cv_sift = cv.SIFT_create(0, 3, contrast_threshold, edge_threshold, 1.6)

    # Get key points and descriptors
    cv_kp, desc = cv_sift.detectAndCompute(im,None)

    return opencv_kp_to_pcv_kp(cv_kp), desc



def plot_features(im,locs,circle=False):
    """ Show image with features. input: im (image as array), 
        locs OpenCv SIFT key points. """

    def draw_circle(c,r):
        t = np.arange(0,1.01,.01)*2*np.pi
        x = r*np.cos(t) + c[1]
        y = r*np.sin(t) + c[0]
        plt.plot(x,y,'b',linewidth=2)

    plt.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2]) 
    else:     
        plt.plot(locs[:,0],locs[:,1],'ob')
    plt.axis('off')


def match(desc1,desc2):
    """ For each descriptor in the first image,
        select its match in the second image.
        input: desc1 (descriptors for the first image),
        desc2 (same for second image). """

    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = np.zeros((desc1_size[0]),'int')
    desc2t = desc2.T # precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:],desc2t) # vector of dot products
        dotprods = 0.9999*dotprods
        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))

        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores


def match_twosided(desc1,desc2,threshold=0.5):
    """ Two-sided symmetric version of match(). """

    matches_12 = match(desc1,desc2)
    matches_21 = match(desc2,desc1)

    # Where returns the values in an array, so the results are listed in an array of
    # length 1, which is why [0] is after the results.
    ndx_12 = np.where(matches_12 >= 0)[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = 0

    return matches_12


def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
    # if none of these cases they are equal, no filling needed.

    return np.concatenate((im1,im2), axis=1)


def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches 
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
        matchscores (as output from 'match()'),
        show_below (if images should be shown below matches). """

    im3 = appendimages(im1,im2)
    if show_below:
        im3 = np.vstack((im3,im3))

    plt.imshow(im3)

    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plt.plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    plt.axis('off')

def opencv_kp_to_pcv_kp(opencv_kp):
    """
    Converts OpenCV key points to the style used by the "Programming Computer Vision" book.
    This was done because I found it a headache to reformat all the OpenCV points when going 
    through later examples.
    """

    # Key points in open CV are (x,y), but (row,col) in the PCV book. So, they are reversed here.
    return np.array([[kp.pt[1], kp.pt[0], kp.size, kp.angle] for kp in opencv_kp])
