"""
Exercise 6

Write a function that matches features between a pair of images and estimates
the scale difference and in-plane rotation of the scene based on the correspondences.
"""

from PIL import Image
import numpy as np
import pylab as plt
from common import imtools
from common import sift

def euclidian_distance(pt1, pt2):
    return np.hypot(pt2[0]-pt1[0],pt2[1]-pt1[1])

def calc_scale_diff(locs1,locs2,matchscores):

    matched_pts1 = []
    matched_pts2 = []

    # First, get the list of matched points.
    for i,m in enumerate(matchscores):

        if m>0:
            matched_pts1.append(locs1[i][:2])
            matched_pts2.append(locs2[m][:2])

    # Calc the distances between the points
    distances1 = []
    distances2 = []

    for i in range(len(matched_pts1)-1):
        distances1.append(euclidian_distance(matched_pts1[i], matched_pts1[i+1]))
        distances2.append(euclidian_distance(matched_pts2[i], matched_pts2[i+1]))

    # average the distances and compute the ratio between the average.
    return np.mean(distances2) / np.mean(distances1)

im1 = np.array(Image.open("data/crans_1_small.jpg").convert("L"))
im2 = np.array(Image.open("data/crans_2_small.jpg").convert("L"))
#im2 = imtools.imresize(im1, [int(im1.shape[1]/2), int(im1.shape[0]/2)])

kp1, desc1 = sift.detect_and_compute(im1)
kp2, desc2 = sift.detect_and_compute(im2)

print('starting matching')
matches = sift.match_twosided(desc1,desc2)

scale_diff = calc_scale_diff(kp1,kp2,matches)

print('Scale Difference: ' + str(scale_diff))
