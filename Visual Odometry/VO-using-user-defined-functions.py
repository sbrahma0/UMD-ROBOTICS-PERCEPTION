#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import random
import math 
import matplotlib.pyplot as plt
import pandas as pd

images = []
path = "stereo/centre/" 
for image in os.listdir(path): # Looping over all the images
    images.append(image) # Storing all the image names in a list
    images.sort() # Sorting the image names

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('model/')
K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]]) # Camera Calibration Matrix of the model

###############################################################################
# Function Name - rotationMatrixToEulerAngles
# Input - R (Rotation Matrix)
# Returns - list of angles in x,y and z axis
###############################################################################

def rotationMatrixToEulerAngles(R) :
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

###############################################################################
# Function Name - fundamentalMatrix
# Input - corners1 (8-features form current image), corners2 (8-features from 
# next image)
# Returns - Fundamental Matrix between the two frames
###############################################################################

def fundamentalMatrix(corners1, corners2):
    A = np.empty((8, 9))

    for i in range(0, len(corners1)): # Looping over all the 8-points (features)
        x1 = corners1[i][0] # x-coordinate from current frame 
        y1 = corners1[i][1] # y-coordinate from current frame
        x2 = corners2[i][0] # x-coordinate from next frame
        y2 = corners2[i][1] # y-coordinate from next frame
        A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    u, s, v = np.linalg.svd(A, full_matrices=True)  # Taking SVD of the matrix
    f = v[-1].reshape(3,3) # Last column of V matrix
    
    u1,s1,v1 = np.linalg.svd(f) 
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) # Constraining Fundamental Matrix to Rank 2
    F = u1 @ s2 @ v1  
    
    return F  

###############################################################################
# Function Name - checkFmatrix
# Input - x1 (feature point from current image), x2 (feature point from next image)
# F (Fundamental Matrix)
# Returns - x2.T * F * x1 
###############################################################################

def checkFmatrix(x1,x2,F): 
    x11=np.array([x1[0],x1[1],1]).T
    x22=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11)))

###############################################################################
# Function Name - calculateEssentialMatrix
# Input - Camera calibration matrix, Fundamental Matrix
# Returns - Essential Matrix 
###############################################################################

def calculateEssentialMatrix(calibrationMatrix, fundMatrix):
    tempMatrix = np.matmul(np.matmul(calibrationMatrix.T, fundMatrix), calibrationMatrix)
    u, s, v = np.linalg.svd(tempMatrix, full_matrices=True)
    sigmaF = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) # Constraining Eigenvalues to 1, 1, 0
    temp = np.matmul(u, sigmaF)
    E_matrix = np.matmul(temp, v)
    return E_matrix

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

###############################################################################
# Function Name - calculatePoseEstimation
# Input - Essential Matrix
# Returns - list of 4 possible rotation matrix, list of 4 possible translation matrix
###############################################################################

def calculatePoseEstimation(essentialMatrix):
    u, s, v = np.linalg.svd(essentialMatrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # 1st Solution
    c1 = u[:, 2] 
    r1 = u @ w @ v
    
    if np.linalg.det(r1) < 0:
        c1 = -c1 
        r1 = -r1
    c1 = c1.reshape((3,1))
    
    # 2nd Solution
    c2 = -u[:, 2]
    r2 = u @ w @ v
    if np.linalg.det(r2) < 0:
        c2 = -c2 
        r2 = -r2 
    c2 = c2.reshape((3,1))
    
    # 3rd Solution
    c3 = u[:, 2]
    r3 = u @ w.T @ v
    if np.linalg.det(r3) < 0:
        c3 = -c3 
        r3 = -r3 
    c3 = c3.reshape((3,1)) 
    
    # 4th Solution
    c4 = -u[:, 2]
    r4 = u @ w.T @ v
    if np.linalg.det(r4) < 0:
        c4 = -c4 
        r4 = -r4 
    c4 = c4.reshape((3,1))
    
    return [r1, r2, r3, r4], [c1, c2, c3, c4]

###############################################################################
# Function Name - getTriangulationPoint
# Input - m1 (Camera Pose from current frame), m2 (Camera Pose from next frame)
# point1 (feature point from current frame), point2 (Corresponding feature point
# from next frame)
# Returns - Triangulated point
###############################################################################

def getTriangulationPoint(m1, m2, point1, point2):
    
    # Skew Symmetric Matrix of point1
    oldx = np.array([[0, -1, point1[1]], [1, 0, -point1[0]], [-point1[1], point1[0], 0]]) 
    # Skew Symmetric Matrix of point2
    oldxdash = np.array([[0, -1, point2[1]], [1, 0, -point2[0]], [-point2[1], point2[0], 0]])
    
    A1 = oldx @ m1[0:3, :] 
    A2 = oldxdash @ m2
    A = np.vstack((A1, A2)) # Ax = 0
    
    u, s, v = np.linalg.svd(A)
    new1X = v[-1]
    new1X = new1X/new1X[3]
    new1X = new1X.reshape((4,1))
    
    return new1X[0:3].reshape((3,1))

###############################################################################
# Function Name - disambiguiousPose
# Input - Rlist (list of rotation matrix), Clist (list of translation vector)
# features1 (Inliers from current frame), features2 (Inliers from next frame)
# Returns - mainr (Disambigious Rotation Matrix), mainc (Disambigious Translation
# vector) 
###############################################################################

def disambiguiousPose(Rlist, Clist, features1, features2):
    check = 0
    Horigin = np.identity(4) # current camera pose is always considered as an identity matrix
    for index in range(0, len(Rlist)): # Looping over all the rotation matrices
        angles = rotationMatrixToEulerAngles(Rlist[index]) # Determining the angles of the rotation matrix
        #print('angle', angles)
        
        # If the rotation of x and z axis are within the -50 to 50 degrees then it is considered down in the pipeline 
        if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50: 
            count = 0 
            newP = np.hstack((Rlist[index], Clist[index])) # New camera Pose 
            for i in range(0, len(features1)): # Looping over all the inliers
                temp1x = getTriangulationPoint(Horigin[0:3,:], newP, features1[i], features2[i]) # Triangulating all the inliers
                thirdrow = Rlist[index][2,:].reshape((1,3)) 
                if np.squeeze(thirdrow @ (temp1x - Clist[index])) > 0: # If the depth of the triangulated point is positive
                    count = count + 1 

            if count > check: 
                check = count
                mainc = Clist[index]
                mainr = Rlist[index]
                
    if mainc[2] > 0:
        mainc = -mainc
                
    #print('mainangle', rotationMatrixToEulerAngles(mainr))
    return mainr, mainc
    
lastH = np.identity(4) # Initial camera Pose is considered as an identity matrix 
origin = np.array([[0, 0, 0, 1]]).T 

l = [] # Variable for storing all the trajectory points

for index in range(19, len(images)-1): # Looping over all the images

    img1 = cv2.imread("stereo/centre/" + str(images[index]), 0) # Read current image
    colorimage1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR) # Convert image from bayer format to BGR format
    undistortedimage1 = UndistortImage(colorimage1,LUT) # Undistort the current image
    gray1 = cv2.cvtColor(undistortedimage1,cv2.COLOR_BGR2GRAY) # Convert the undistorted image in grayscale
    
    img2 = cv2.imread("stereo/centre/" + str(images[index + 1]), 0) # Read next image
    colorimage2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR) # Convert image from bayer format to BGR format
    undistortedimage2 = UndistortImage(colorimage2,LUT) # Undistort the next image
    gray2 = cv2.cvtColor(undistortedimage2,cv2.COLOR_BGR2GRAY) # Convert the undistorted image in grayscale
    
    gray1_crop_img = gray1[200:650, 0:1280] # Cropping the area of interest from the current frame
    gray2_crop_img = gray2[200:650, 0:1280] # Cropping the area of interest from the next frame

    sift = cv2.xfeatures2d.SIFT_create() 

    # find the keypoints and descriptors with SIFT in current as well as next frame
    kp1, des1 = sift.detectAndCompute(gray1_crop_img,None)
    kp2, des2 = sift.detectAndCompute(gray2_crop_img,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    features1 = [] # Variable for storing all the required features from the current frame
    features2 = [] # Variable for storing all the required features from the next frame

    # Ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            features1.append(kp1[m.queryIdx].pt)
            features2.append(kp2[m.trainIdx].pt)
        
    noOfInliers = 0
    finalFundMatrix = np.zeros((3,3))

    inlier1 = [] # Variable for storing all the inliers features from the current frame
    inlier2 = [] # Variable for storing all the inliers features from the next frame
    # RANSAC Algorithm
    for i in range(0, 50): # 50 iterations for RANSAC 
        count = 0
        eightpoint = [] 
        goodFeatures1 = [] # Variable for storing eight random points from the current frame
        goodFeatures2 = [] # Variable for storing corresponding eight random points from the next frame
        tempfeature1 = [] 
        tempfeature2 = []
        
        while(True): # Loop runs while we do not get eight distinct random points
            num = random.randint(0, len(features1)-1)
            if num not in eightpoint:
                eightpoint.append(num)
            if len(eightpoint) == 8:
                break

        for point in eightpoint: # Looping over eight random points
            goodFeatures1.append([features1[point][0], features1[point][1]]) 
            goodFeatures2.append([features2[point][0], features2[point][1]])
    
        # Computing Fundamentals Matrix from current frame to next frame
        FundMatrix = fundamentalMatrix(goodFeatures1, goodFeatures2)

        for number in range(0, len(features1)):
            
            # If x2.T * F * x1 is less than threshold (0.01) then it is considered as Inlier
            if checkFmatrix(features1[number], features2[number], FundMatrix) < 0.01:
                count = count + 1 
                tempfeature1.append(features1[number])
                tempfeature2.append(features2[number])

        if count > noOfInliers: 
            noOfInliers = count
            finalFundMatrix = FundMatrix
            inlier1 = tempfeature1
            inlier2 = tempfeature2

    # Computing Essential Matrix from current frame to next frame
    essentialMatrix = calculateEssentialMatrix(K, finalFundMatrix)
    # Computing all the solutions of rotation matrix and translation vector
    Rlist, Tlist = calculatePoseEstimation(essentialMatrix)
    # Disambiguating one solution from four
    R, T = disambiguiousPose(Rlist, Tlist, inlier1, inlier2) 

    lastH = lastH @ Homogenousmatrix(R, T) # Transforming from current frame to next frame
    p = lastH @ origin # Determining the transformation of the origin from current frame to next frame

    #print('x- ', p[0])
    #print('y- ', p[2])
    l.append([p[0][0], -p[2][0]])
    plt.scatter(p[0][0], -p[2][0], color='r')
    
    if cv2.waitKey(0) == 27:
        break
        
cv2.destroyAllWindows()
#df = pd.DataFrame(l, columns = ['X', 'Y'])
#df.to_excel('test_code_last1.xlsx')
plt.show()


# In[ ]:




