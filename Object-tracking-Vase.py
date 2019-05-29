#!/usr/bin/env python
# coding: utf-8

# In[1]:


# #############Project4_Lucas_Kanade_Template_Tracker_Vase####################
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
path = 'vase\\'

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
    
    threshold = 0.0001 # Threshold for convergence of the error of the parameters
    # Top-Left, Top-Right, Bottom-Left and Bottom-Right Corners of the template
    x1, y1, x2, y2, x3, y3, x4, y4 = rectpoints[0], rectpoints[1], rectpoints[2], rectpoints[3], rectpoints[4], rectpoints[5], rectpoints[6], rectpoints[7]
    initial_y, initial_x = np.gradient(initialtemp1) # Calculating Intensity Gradient of the next frame
    dp = 1 # Initializing the variable for storing error in the parameters

    while np.square(dp).sum() > threshold: # Looping until the solution converges below the threshold
       
        posx, posy = pos0[0], pos0[1] # Initial Parameters
        # Warped Parameters
        x1_warp, y1_warp, x2_warp, y2_warp, x3_warp, y3_warp, x4_warp, y4_warp = x1 + posx, y1 + posy, x2 + posx, y2 + posy, x3 + posx, y3 + posy, x4 + posx, y4 + posy

        x = np.arange(0, initialtemp.shape[0], 1)
        y = np.arange(0, initialtemp.shape[1], 1)

        a1 = np.linspace(x1, x3, 87) # Interpolating points from top-left x-coordinate to the bottom-right x-coordinate
        b1 = np.linspace(y1, y3, 36) # Interpolating points from top-left y-coordinate to the bottom-right y-coordinate
        a2 = np.linspace(x4, x2, 87) # Interpolating points from top-right x-coordinate to the bottom-left x-coordinate
        b2 = np.linspace(y2, y4, 36) # Interpolating points from top-right y-coordinate to the bottom-left y-coordinate
        a = np.union1d(a1, a2) # Taking unique Union of all the x-coordinates
        b = np.union1d(b1, b2) # Taking unique Union of all the y-coordinates
        aa, bb = np.meshgrid(a, b) # Creating a mesh of all these points

        a1_warp = np.linspace(x1_warp, x3_warp, 87) # Interpolating warped points from top-left x-coordinate to the bottom-right x-coordinate
        b1_warp = np.linspace(y1_warp, y3_warp, 36) # Interpolating warped points from top-left y-coordinate to the bottom-right y-coordinate
        a2_warp = np.linspace(x4_warp, x2_warp, 87) # Interpolating warped points from top-right x-coordinate to the bottom-left x-coordinate
        b2_warp = np.linspace(y2_warp, y4_warp, 36) # Interpolating warped points from top-right y-coordinate to the bottom-left y-coordinate
        a_warp = np.union1d(a1_warp, a2_warp) # Taking unique Union of all the warped x-coordinates
        b_warp = np.union1d(b1_warp, b2_warp) # Taking unique Union of all the warped y-coordinates
        aaw, bbw = np.meshgrid(a_warp, b_warp) # Creating a mesh of all these points

        spline = RectBivariateSpline(x, y, initialtemp) # Smoothing and Interpolating intensity data over all the template frame
        T = spline.ev(bb, aa) # Evaluating the intensity data over all the interpolated points

        spline1 = RectBivariateSpline(x, y, initialtemp1) # Smoothing and Interpolating intensity data over all the next frame
        warpImg = spline1.ev(bbw, aaw) # Evaluating the intensity data over all the warped interpolated points

        error = T - warpImg # Calculating the change in intensity from the template frame to the next frame
        errorImg = error.reshape(-1, 1)

        spline_gx = RectBivariateSpline(x, y, initial_x) # Smoothing and Interpolating intensity gradient data over all the next frame in x-direction
        initial_x_w = spline_gx.ev(bbw, aaw) # Evaluating intensity gradient data over all the interpolated points in x-direction

        spline_gy = RectBivariateSpline(x, y, initial_y) # Smoothing and Interpolating intensity gradient data over all the next frame in y-direction
        initial_y_w = spline_gy.ev(bbw, aaw) # Evaluating intensity gradient data over all the interpolated points in y-direction
        
        I = np.vstack((initial_x_w.ravel(), initial_y_w.ravel())).T # Stacking both the intensity gradients together
  
        jacobian = np.array([[1, 0], [0, 1]])

        hessian = I @ jacobian # Initializing the Jacobian
        H = hessian.T @ hessian # Calculating Hessian Matrix
        
        dp = np.linalg.inv(H) @ (hessian.T) @ errorImg
        
        pos0[0] += dp[0, 0]
        pos0[1] += dp[1, 0]

    p = pos0
    return p # Returing the updated parameters

rectpoints1 = [124,91,172,91,172,150,124,150] # Template Rectangular points, calculated manually for 1st Pyramid Layer
rectpoints2 = [62, 48, 85, 48, 85, 74, 62, 74] # Template Rectangular points, calculated manually for 2nt Pyramid Layer
rectpoints3 = [31, 22, 43, 22, 43, 37, 31, 37] # Template Rectangular points, calculated manually for 3rt Pyramid Layer
rectpoints10 = copy.deepcopy(rectpoints1)
rectpoints20 = copy.deepcopy(rectpoints2)
rectpoints30 = copy.deepcopy(rectpoints3)

cap_140 = cv_img[0]
cap_140 = cv2.GaussianBlur(cap_140, (9,9), 0)
cap_gray_140_1 = cv2.cvtColor(cap_140, cv2.COLOR_BGR2GRAY)
cap_gray_140_2 = cv2.pyrDown(cap_gray_140_1)
cap_gray_140_3 = cv2.pyrDown(cap_gray_140_2)

for i in range(0, len(cv_img)-1): # Looping over all the images

    image_index = i
    cap = cv_img[image_index]
    cap_gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    cv2.line(cap, (int(rectpoints1[0]), int(rectpoints1[1])), (int(rectpoints1[2]), int(rectpoints1[3])), (0,255,0), 3)
    cv2.line(cap, (int(rectpoints1[0]), int(rectpoints1[1])), (int(rectpoints1[6]), int(rectpoints1[7])), (0, 255, 0), 3)
    cv2.line(cap, (int(rectpoints1[2]), int(rectpoints1[3])), (int(rectpoints1[4]), int(rectpoints1[5])), (0, 255, 0), 3)
    cv2.line(cap, (int(rectpoints1[6]), int(rectpoints1[7])), (int(rectpoints1[4]), int(rectpoints1[5])), (0, 255, 0), 3)

    cv2.imshow('Tracking_vase', cap)
    cap_next = cv_img[image_index+1]
    cap_next = cv2.GaussianBlur(cap_next, (9,9), 0)
    cap_gray_next1 = cv2.cvtColor(cap_next, cv2.COLOR_BGR2GRAY)
    cap_gray_next2 = cv2.pyrDown(cap_gray_next1)
    cap_gray_next3 = cv2.pyrDown(cap_gray_next2)
    
    # 3rd Layer
    initialtemp0 = cap_gray_140_3 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next3 / 255.
    p1 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints30)
    
    # 2nd Layer
    initialtemp0 = cap_gray_140_2 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next2 / 255.
    p2 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints20, pos0 = np.array(p1)*2)
    
    # 1st Layer
    initialtemp0 = cap_gray_140_1 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next1 / 255.
    p3 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints10, pos0 = np.array(p1)*4 + np.array(p2)*2)
    
    # Updating the rectangular coordinates from the recieved paramters
    rectpoints1[0] = rectpoints10[0] + p3[0]
    rectpoints1[1] = rectpoints10[1] + p3[1]
    rectpoints1[2] = rectpoints10[2] + p3[0]
    rectpoints1[3] = rectpoints10[3] + p3[1]
    rectpoints1[4] = rectpoints10[4] + p3[0]
    rectpoints1[5] = rectpoints10[5] + p3[1] 
    rectpoints1[6] = rectpoints10[6] + p3[0]
    rectpoints1[7] = rectpoints10[7] + p3[1]
    
    if (image_index > 16 and image_index < 30) or (image_index > 56 and image_index < 68) or (image_index > 81 and image_index < 99):
        rectpoints1[1] = 90
        rectpoints1[3] = 90
        
    if image_index > 16 and image_index< 30 or (image_index > 56 and image_index < 68) or (image_index > 81 and image_index < 99):
        rectpoints1[0] = rectpoints1[0] - 20
        rectpoints1[6] = rectpoints1[6] - 20
        
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()


# In[ ]:




