#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import copy
from numpy.linalg import matrix_rank
import pandas as pd

###############################################################################
# Function Name - getCameraMatric
# Input - path of the file
# Returns - Camera Calibration Matrix and LUT Transform
###############################################################################

def getCameraMatric(path):
    images = []
    for image in os.listdir(path):
        images.append(image)
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('model/')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K, LUT

###############################################################################
# Function Name - undistortImageToGray
# Input - img (image in bayer format)
# Returns - image converted into gray format
###############################################################################

def undistortImageToGray(img):
    colorimage = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistortedimage = UndistortImage(colorimage, LUT)
    gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)

    return gray

###############################################################################
# Function Name - returnPose
# Input - und1 (current frame in gray format), und2 (nextframe in gray format)
# Returns - translation and orientation of next frame with respect to current frame
###############################################################################

def returnPose(und1, und2, k):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(und1, None)
    kp2, des2 = sift.detectAndCompute(und2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 1 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    E = k.T @ F @ k
    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, k)
    return R, t

###############################################################################
# Function Name - Homogenousmatrix
# Input - R (Rotation Matrix), t (translation Matrix)
# Returns - A Homogeneous Matrix
###############################################################################

def Homogenousmatrix(R, t):
    z = np.column_stack((R, t))
    a = np.array([0, 0, 0, 1])
    z = np.vstack((z, a))
    return z


sift = cv2.xfeatures2d.SIFT_create()
path = "stereo\\centre\\"
k, LUT = getCameraMatric(path)

images = []
for image in os.listdir(path):
    images.append(image)

h1 = np.identity(4)
t1 = np.array([[0, 0, 0, 1]])
t1 = t1.T

l = []

for index in range(19, len(images) - 1):  # -1 is most important
    img1 = cv2.imread("%s\\%s" % (path, images[index]), 0)
    und1 = undistortImageToGray(img1)

    img2 = cv2.imread("%s\\%s" % (path, images[index + 1]), 0)
    und2 = undistortImageToGray(img2)

    R, T = returnPose(und1, und2, k)

    h2 = Homogenousmatrix(R, T)
    h1 = h1 @ h2
    p = h1 @ t1

    #print('x- ', p[0])
    #print('y- ', p[2])

    plt.scatter(p[0][0], -p[2][0], color='r')
    l.append([p[0][0], -p[2][0]])

    #print('loop', index - 1)

#df = pd.DataFrame(l, columns = ['X', 'Y']) # Remove hash to store data in an excel sheet
#df.to_excel('test.xlsx')
plt.show()


# In[ ]:




