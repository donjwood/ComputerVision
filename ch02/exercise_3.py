"""
Exercise 3

An alternative corner detector to Harris is the FAST corner detector. There
are a number of implementations including a pure Python version available at
http://www.edwardrosten.com/work/fast.html. Try this detector, play with
the sensitivity threshold, and compare the corners with the ones from our Harris
implementation.

Using the OpenCV module. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html
Documentation is not up to date, here are the correct calls:
https://stackoverflow.com/questions/37206077/where-is-the-fast-algorithm-in-opencv 
"""
import cv2
import pylab as plt

threshold = 10

img = cv2.imread('data/empire.jpg', 0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=threshold)

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))

plt.figure('FAST Corner Detection Key Points')
plt.imshow(img2)
plt.show()
