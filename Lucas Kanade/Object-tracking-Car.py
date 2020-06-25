#!/usr/bin/env python
# coding: utf-8

# In[3]:


# #############Project4_Lucas_Kanade_Template_Tracker_Car#####################
# Team Members (Group Name - Sayan+Nikhil+Pranali)
# PRANALI DESAI - 116182935
# NIKHIL MEHRA - 116189941
# SAYAN BRAHMA - 116309165
##############################################################################

import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import os

images = []
cv_img = []
path = 'car\\'

for image in os.listdir(path): # Looping over all the car images and storing them in one list
    images.append(image)
images.sort()

for image in images:
    img = cv2.imread("%s%s" % (path, image))
    cv_img.append(img)

##############################################################################
# Function Name - warpInverse 
# Arguments - Affine Parameters
# Returns - Lucas-Kanade Inverse Compositional Parameters
##############################################################################

def warpInverse(p):
    inverse_output = [0, 0, 0, 0, 0, 0]
    deno = (1 + p[0]) * (1 + p[3]) - p[1] * p[2]
    inverse_output[0] = (-p[0] - p[0] * p[3] + p[1] * p[2]) / deno
    inverse_output[1] = (-p[1]) / deno
    inverse_output[2] = (-p[2]) / deno
    inverse_output[3] = (-p[3] - p[0] * p[3] + p[1] * p[2]) / deno
    inverse_output[4] = (-p[4] - p[3] * p[4] + p[2] * p[5]) / deno
    inverse_output[5] = (-p[5] - p[0] * p[5] + p[1] * p[4]) / deno
    return inverse_output

##############################################################################
# Function Name - LucasKanadefunction 
# Arguments - initialtemp(frame which contains the template to track),
# initialtemp1(frame in which you need to track the template), rectpoints(list
# of top-left and bottom-right corners of the rectangular template), 
# pos0(parameters for the Lucas Kanade Function initialized with zeros initially)
# Returns - Updates parameters
# Algorithm - Uses Lucas Kanade Template Tracker Algorithm for tracking a manually 
# defined tracker from initialtemp frame to initialtemp1 frame.
##############################################################################

def LucasKanadefunction(initialtemp, initialtemp1, rectpoints, pos0=np.zeros(2)):
    
    threshold = 0.001 # Threshold for convergence of the error of the parameters
    x1, y1, x2, y2 = rectpoints[0], rectpoints[1], rectpoints[2], rectpoints[3] # Top-Left and Bottom-Right Corners of the template
    initial_y, initial_x = np.gradient(initialtemp1) # Calculating Intensity Gradient of the next frame
    dp = 1 # Initializing the variable for storing error in the parameters

    while np.square(dp).sum() > threshold: # Looping until the solution converges below the threshold
    
        posx, posy = pos0[0], pos0[1] # Initial Parameters
        x1_warp, y1_warp, x2_warp, y2_warp = x1 + posx, y1 + posy, x2 + posx, y2 + posy # Warped Parameters

        x = np.arange(0, initialtemp.shape[0], 1)
        y = np.arange(0, initialtemp.shape[1], 1)

        a = np.linspace(x1, x2, 87) # Interpolating points from top-left x-coordinate to the bottom-right x-coordinate
        b = np.linspace(y1, y2, 36) # Interpolating points from top-left y-coordinate to the bottom-right y-coordinate
        aa, bb = np.meshgrid(a, b) # Creating a mesh of all these points

        a_warp = np.linspace(x1_warp, x2_warp, 87) # Interpolating warped points from top-left x-coordinate to the bottom-right x-coordinate
        b_warp = np.linspace(y1_warp, y2_warp, 36) # Interpolating warped points from top-left y-coordinate to the bottom-right y-coordinate
        aa_warp, bb_warp = np.meshgrid(a_warp, b_warp) # Creating a mesh of all these warped points

        spline = RectBivariateSpline(x, y, initialtemp) # Smoothing and Interpolating intensity data over all the template frame
        T = spline.ev(bb, aa) # Evaluating the intensity data over all the interpolated points

        spline1 = RectBivariateSpline(x, y, initialtemp1) # Smoothing and Interpolating intensity data over all the next frame
        warpImg = spline1.ev(bb_warp, aa_warp) # Evaluating the intensity data over all the warped interpolated points

        error = T - warpImg  # Calculating the change in intensity from the template frame to the next frame
        errorImg = error.reshape(-1, 1)

        spline_gx = RectBivariateSpline(x, y, initial_x) # Smoothing and Interpolating intensity gradient data over all the next frame in x-direction
        initial_x_warp = spline_gx.ev(bb_warp, aa_warp) # Evaluating intensity gradient data over all the interpolated points in x-direction

        spline_gy = RectBivariateSpline(x, y, initial_y) # Smoothing and Interpolating intensity gradient data over all the next frame in y-direction
        initial_y_warp = spline_gy.ev(bb_warp, aa_warp) # Evaluating intensity gradient data over all the interpolated points in y-direction
        
        I = np.vstack((initial_x_warp.ravel(), initial_y_warp.ravel())).T # Stacking both the intensity gradients together

        jacobian = np.array([[1, 0], [0, 1]])

        hessian = I @ jacobian # Initializing the Jacobian
        H = hessian.T @ hessian # Calculating Hessian Matrix

        dp = np.linalg.inv(H) @ (hessian.T) @ errorImg # Calculating the change in parameters
        # Updating the previous parameters
        pos0[0] += dp[0, 0]
        pos0[1] += dp[1, 0]

    p = pos0
    return p # Returing the updated parameters

rectpoints = [118,100, 338,280] # Template Rectangular points, calculated manually
width = rectpoints[3] - rectpoints[1] # Calculating the width of the template
length = rectpoints[2] - rectpoints[0] # Calculating the length of the template
rectpoints0 = copy.deepcopy(rectpoints)
cap_orig = cv_img[0]
cap_gray_orig = cv2.cvtColor(cap_orig, cv2.COLOR_BGR2GRAY)
cap_gray_orig = cv2.equalizeHist(cap_gray_orig)

for i in range(0, len(cv_img)-1): # Looping over all the images

    image_index = i

    if image_index == 100:
        rectpoints = [145, 113, 345, 275]
        width = rectpoints[3] - rectpoints[1]
        length = rectpoints[2] - rectpoints[0]
        rectpoints0 = copy.deepcopy(rectpoints)
        cap_orig = cv_img[100]
        cap_gray_orig = cv2.cvtColor(cap_orig, cv2.COLOR_BGR2GRAY)
        cap_gray_orig = cv2.equalizeHist(cap_gray_orig)

    if image_index == 150:
        rectpoints = [200,113,395,275]
        width = rectpoints[3] - rectpoints[1]
        length = rectpoints[2] - rectpoints[0]
        rectpoints0 = copy.deepcopy(rectpoints)
        cap_orig = cv_img[150]
        cap_gray_orig = cv2.cvtColor(cap_orig, cv2.COLOR_BGR2GRAY)
        cap_gray_orig = cv2.equalizeHist(cap_gray_orig)

    if image_index == 200:
        rectpoints = [295,125,446,240]
        width = rectpoints[3] - rectpoints[1]
        length = rectpoints[2] - rectpoints[0]
        rectpoints0 = copy.deepcopy(rectpoints)
        cap_orig = cv_img[200]
        cap_gray_orig = cv2.cvtColor(cap_orig, cv2.COLOR_BGR2GRAY)
        cap_gray_orig = cv2.equalizeHist(cap_gray_orig)

    cap = cv_img[image_index]
    maincap= cap
    cap_gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    cap_gray = cv2.equalizeHist(cap_gray) # Converting image into the grayscale
    # Drawing rectangle over the object to track
    cv2.rectangle(maincap,(int(rectpoints[0]),int(rectpoints[1])),(int(rectpoints[0])+length,int(rectpoints[1])+width),(0,255,0),3)
    cv2.imshow('Car_Tracking', cap) # Showing output of the tracking

    cap_next = cv_img[image_index+1]
    cap_gray_next = cv2.cvtColor(cap_next, cv2.COLOR_BGR2GRAY)
    cap_gray_next = cv2.equalizeHist(cap_gray_next)

    initialtemp0 = cap_gray_orig / 255.
    initialtemp1 = cap_gray_next / 255.
    
    p = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints0) # Calling Lucas Kanade Function
    # Updating the rectangular coordinates from the recieved paramters
    rectpoints[0] = rectpoints0[0] + p[0]
    rectpoints[1] = rectpoints0[1] + p[1]
    rectpoints[2] = rectpoints0[2] + p[0]
    rectpoints[3] = rectpoints0[3] + p[1]

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break

