from PIL import Image
import cv2 as cv
import numpy as np
import pylab as plt
from common import imtools

im1 = np.array(Image.open("data/crans_1_small.jpg").convert("L"))
im2 = np.array(Image.open("data/crans_2_small.jpg").convert("L"))

sift = cv.SIFT_create()

# Get key points and descriptors
kp1, desc1 = sift.detectAndCompute(im1,None)
kp2, desc2 = sift.detectAndCompute(im2,None)

# Match the points
bf = cv.BFMatcher()
matches = bf.knnMatch(desc1,desc2,k=2)


# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

im3 = cv.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure("SIFT Matching")
plt.imshow(im3)

plt.show()
