"""
Exercise 5

The VLFeat command line tools also contain an implementation of Maximally
Stable Extremal Regions (MSER), http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions,
a region detector that finds blob like regions. Create
a function for extracting MSER regions and pass them to the descriptor part of
SIFT using the "--read-frames" option and one function for plotting the ellipse
regions.

Helpful Links:
https://stackoverflow.com/questions/40078625/opencv-mser-detect-text-areas-python
https://stackoverflow.com/questions/53918522/ellipse-fitting-for-images-using-fitellipse
"""

import cv2

img = cv2.imread('data/empire.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = gray.copy()
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(img)
for p in regions:
    ellipse = cv2.fitEllipse(p)
    cv2.ellipse(vis, ellipse, (0, 255, 0), 1, cv2.LINE_AA)
cv2.imshow('img', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
