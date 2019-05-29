#!/usr/bin/env python
# coding: utf-8

# In[97]:


# #####################Project6_Traffic_Detection ############################
# Team Members (Group Name - Sayan+Nikhil+Pranali)
# PRANALI DESAI - 116182935
# NIKHIL MEHRA - 116189941
# SAYAN BRAHMA - 116309165
##############################################################################

import numpy as np
import cv2
import os
import copy
from PIL import Image
from skimage.feature import hog
from skimage import feature, exposure
from sklearn import svm
from scipy.misc import imread

# Change input image folder below 
path = "input\\" 

frames = []
# Looping over all the input images 
for frame in os.listdir(path): 
    frames.append(frame)
    frames.sort()

# Training set for blue signs
train_set = [["Training\\00035", 35], ["Training\\00038", 38], ["Training\\00045", 45]]

###############################################################################################
#####################################TrainingBlue##############################################
hog_list = []
label_list = []
count = 0

# Looping over all the blue training set data
for name in train_set:
    value = name[0]
    label = name[1]
    image_list = [os.path.join(value, f) for f in os.listdir(value) if f.endswith('.ppm')]
    for image in range(0, len(image_list)):
        count += 1
        im = np.array(Image.open(image_list[image]))
        im_prep = cv2.resize(im, (16, 16))
        # Extracting HOG Features
        fd, h = feature.hog(im_prep, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1", visualise=True, multichannel=True)
        hog_list.append(fd)
        label_list.append(label)
        
clf_b = svm.SVC(gamma='scale', decision_function_shape='ovo')
# Blue Classifier
clf_b.fit(hog_list, label_list)
###############################################################################################

# Training set for red signs
train_set = [["Training\\00001", 1], ["Training\\00014", 14], ["Training\\00017", 17], ["Training\\00019", 19], 
                ["Training\\00021", 21]]

################################################################################################
#####################################TrainingRed################################################
hog_list = []
label_list = []
count = 0

# Looping over all the red training data
for name in train_set:
    value = name[0]
    label = name[1]
    image_list = [os.path.join(value, f) for f in os.listdir(value) if f.endswith('.ppm')]
    for image in range(0, len(image_list)): 
        count += 1
        im = np.array(Image.open(image_list[image]))
        im_prep = cv2.resize(im, (16, 16))
        # Extracting HOG Features
        fd, h = feature.hog(im_prep, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1", visualise=True, multichannel=True)
        hog_list.append(fd)
        label_list.append(label)
        
clf_r = svm.SVC(gamma='scale', decision_function_shape='ovo')
# Red Classifier
clf_r.fit(hog_list, label_list)
#############################################################################################


# In[105]:


# out = cv2.VideoWriter('outpy3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (800, 600))

###############################################################################
# Function Name - Classify_b
# Input - image (Detected Sign Image)
# Returns - Classified result image from the blue classifier
# Algorithm - Uses SVM Classifier 
###############################################################################

def Classify_b(image):
    
    # Resizing into the same size as training data
    image1 = cv2.resize(image, (16, 16))
    hog_list_test = []
    # Extracting HOG Features
    fd, h = feature.hog(image1, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1", visualise=True, multichannel=True)
    hog_list_test.append(fd)
    
    predictions = []
    # Making predictions through SVM classifier(blue)
    predictions = clf_b.predict(hog_list_test)
    if predictions[0] in [1, 17, 14, 19, 21, 35, 38, 45]:
        result = cv2.imread('Result/'+str(predictions[0])+'.PNG')
    
    return result

###############################################################################
# Function Name - Classify_r
# Input - image (Detected Sign Image)
# Returns - Classified result image from the red classifier
# Algorithm - Uses SVM Classifier
###############################################################################

def Classify_r(image):
    
    # Resizing into the same size as training data
    image1 = cv2.resize(image, (16, 16))        
    hog_list_test = []
    # Extracting HOG Features
    fd, h = feature.hog(image1, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1", visualise=True, multichannel=True)
    hog_list_test.append(fd)
    
    predictions = []
    # Making predictions through SVM classifier(blue)
    predictions = clf_r.predict(hog_list_test)
    if predictions[0] in [1, 17, 14, 19, 21, 35, 38, 45]:
        result = cv2.imread('Result/'+str(predictions[0])+'.PNG')
    
    return result

###############################################################################
# Function Name - findBoundingBox
# Input - contours(detected contours through MSER), color(color being detected 
# through MSER)
# Returns - maximum area bounding box corresponding each sign present in the frame
# Algorithm - Group all contours by centroid displacement and then finding the 
# maximum area contour among them
###############################################################################
    
def findBoundingBox(contours, color):
    centerdict = {} # Varibale to store different groups of contours
    
    # Looping over all the contours
    for i in range(0, len(contours)):
        M = cv2.moments(contours[i]) # Finding moments of the contour
        
        if M["m00"] != 0:
            # Detecting the centroid of the contour
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            if i == 0:
                centerdict[(cX, cY)] = [i]
            else:
                flag = 0
                for key in list(centerdict.keys()):
                    # Threshold distance for grouping is 200 pixels in each direction
                    if (cX - key[0])**2 + (cY - key[1])**2 - 200**2 < 0:
                        centerdict[key].append(i)
                        flag = 1
                        break
                if flag == 0:
                    centerdict[(cX, cY)] = [i]
                    
    boxes = [] 
    # Looping over all the groups
    for key in list(centerdict.keys()):
        temp = 0
        # Rejecting groups with less than and equal to 3 contours
        if len(centerdict[key]) > 3 and color == 'b':
            for index in centerdict[key]:
                area = cv2.contourArea(contours[index])
                if area > temp:
                    temp = area
                    main = contours[index]
            boxes.append(main)
        elif color == 'r':
            for index in centerdict[key]:
                area = cv2.contourArea(contours[index])
                if area > temp:
                    temp = area
                    main = contours[index]
            boxes.append(main)
            
    return boxes # Returns list with the maximum area bounding box about each sign
                                       
for index in range(0,len(frames)): # Looping over all the frames
    
    img = cv2.imread("input/" + str(frames[index])) # Reading Frames
    img1 = cv2.resize(img, (800,600), interpolation = cv2.INTER_AREA) # Resizing Frames
    dst = cv2.fastNlMeansDenoisingColored(img1, None,10,10,7,21) # Denoising Frames

    # Splitting all frames in their respective channel
    bluec = dst[:,:,0] # Blue channel
    greenc = dst[:,:,1] # Green channel
    redc = dst[:,:,2] # Red channel

    # Contrast Normalization of the blue channel
    minmax_img_b = bluec - np.min(bluec)
    minmax_img_b = minmax_img_b/(np.max(bluec)-np.min(bluec))
    minmax_img_b = minmax_img_b * 255

    # Contrast Normalization of the green channel
    minmax_img_g = greenc - np.min(greenc)
    minmax_img_g = minmax_img_g/(np.max(greenc)-np.min(greenc))
    minmax_img_g = minmax_img_g * 255
    
    # Contrast Normalization of the red channel
    minmax_img_r = redc - np.min(redc)
    minmax_img_r = minmax_img_r/(np.max(redc)-np.min(redc))
    minmax_img_r = minmax_img_r * 255

    # A Zero Numpy array for comparing
    zero_img = np.zeros((img1.shape[0],img1.shape[1]))

    # Normalizing Intensity for blue color Detection
    num = minmax_img_b - minmax_img_r
    den = minmax_img_b + minmax_img_g + minmax_img_r
    total = num/den
    total = np.where(np.invert(np.isnan(total)), total, 0)
    normalize_img_b = (np.maximum(zero_img, total)*255).astype(np.uint8)
    normalize_img_b = np.where(normalize_img_b  > 45 , normalize_img_b, 0)
    normalize_img_b = np.where(normalize_img_b  < 150 , normalize_img_b, 0)
    
    # Normalizing Intensity for red color Detection
    num1 = minmax_img_r - minmax_img_b
    num2 = minmax_img_r - minmax_img_g
    den1 = minmax_img_b + minmax_img_g + minmax_img_r
    total1 = np.minimum(num1, num2)/den1
    total1 = np.where(np.invert(np.isnan(total1)), total1, 0)
    normalize_img_r = (np.maximum(zero_img, total1)*255).astype(np.uint8)
    normalize_img_r = np.where(normalize_img_r > 10, normalize_img_r, 0)
    normalize_img_r = np.where(normalize_img_r < 90, normalize_img_r, 0)

    # Initializing MSER for blue
    bmser = cv2.MSER_create(2, 100, 1000, 0.3, 0.2, 200, 1.01, 0.003, 5)
    # Initializing MSER for red
    rmser = cv2.MSER_create(20, 100, 1000, 1.2, 0.2, 200, 1.01, 0.003, 5)
    
    imagecopy = img1.copy()
    # Detecting areas on normalized blue image using MSER
    regions, _ = bmser.detectRegions(normalize_img_b)
    # Finding maximum area using findBounding Box Function
    blueregions =  findBoundingBox(regions, 'b')
    
    for region in blueregions: # Looping over all the good areas
        x,y,w,h = cv2.boundingRect(region) # Finding bounding box over the area
        # If the top left corner is in the upper 80% of the image and the aspect ratio is less than 1.1
        if  y < 200 and (w/h) < 1.1 :  
            # Increasing the bounding box area for proper classification
            if x - 5 > 0 and y - 5 > 0:
                x = x - 5
                y = y - 5
                w = w + 10
                h = h + 10
            # Drawing the bounding boxes  
            cv2.rectangle(imagecopy,(x,y),(x+w,y+h),(0,255,0),2)
            # Cropping the detected bounding box for classification
            testimage = imagecopy[y:y+h, x:x+w]
            # Classifying the crop image
            resultimage = Classify_b(testimage)
            # Resizing the resulted classified image 
            resultimageresized = cv2.resize(resultimage, (w, h))
        
            if x-w > 0: # Pasting the resulted image besides the detected boundary
                imagecopy[y:y+h, x-w:x] = resultimageresized
            else:
                imagecopy[y:y+h, x+w:x+2*w] = resultimageresized
    
    # Detecting areas on normalized red image using MSER
    regions1, _ = rmser.detectRegions(normalize_img_r)
    # Finding maximum area using findBounding Box Function
    redregions =  findBoundingBox(regions1, 'r') 
    
    for region1 in redregions: # Looping over all the good areas
        area = cv2.contourArea(region1) # Finding bounding box over the area
        x1,y1,w1,h1 = cv2.boundingRect(region1)
        # If the top left corner is in the upper right side of the image and the aspect ratio is less than 1.1 and greater than 0.2
        # If the patch area is greater than 170
        if y1 < 90 and x1> 400  and (w1/h1) <= 1.1 and area > 170 and (w1/h1) >= 0.2:
            # Increasing the bounding box area for proper classification
            if x1 - 5 > 0 and y1 - 5 > 0: 
                x1 = x1 - 5
                y1 = y1 - 5
                w1 = w1 + 10
                h1 = h1 + 10
            # Drawing the bounding boxes  
            cv2.rectangle(imagecopy,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
            # Cropping the detected bounding box for classification
            testimager = imagecopy[y1:y1+h1, x1:x1+w1]
            # Classifying the crop image
            resultimager = Classify_r(testimager)
            # Resizing the resulted classified image 
            resultimageresizedr = cv2.resize(resultimager, (w1, h1))
            
            if x1-w1 > 0: # Pasting the resulted image besides the detected boundary
                imagecopy[y1:y1+h1, x1-w1:x1] = resultimageresizedr
            else:
                imagecopy[y1:y1+h1, x1+w1:x1+2*w1] = resultimageresizedr
            
    cv2.imshow('Sign Detection', imagecopy) # Showing the detected signs
    #out.write(imagecopy)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
#out.release()


# In[ ]:




