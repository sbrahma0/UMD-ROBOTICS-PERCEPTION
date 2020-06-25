#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib import style
import pylab
from pandas import ExcelWriter
from pandas import ExcelFile
import cv2
import os
import numpy as np
import os

images = []
path = "stereo\\centre\\"
df = pd.read_excel('test.xlsx', sheet_name='Sheet1')
df2 = pd.read_excel('test_code.xlsx', sheet_name='Sheet1')
for image in os.listdir("stereo\\centre\\"):
    images.append(image)
sift = cv2.xfeatures2d.SIFT_create()
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('model/')
#fig1 = plt.figure()


for i in df.index:
    print(i)
    print(df['X'][i], df['Y'][i])
    print(df2['X'][i], df2['Y'][i])
    img1 = mpimg.imread("%s\\%s" % (path, images[19+i]), 0)
    img = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    undistortedimage = UndistortImage(img, LUT)
    gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    img3 = cv2.drawKeypoints(gray, kp, img,  (255,0,0))
    newX, newY = img3.shape[1] * 2, img3.shape[0] * 1
    newimg = cv2.resize(img3, (int(newX), int(newY)))


    plt.subplot(224)
    plt.xlabel('FRAMES', fontsize=10)
    plt.ylabel('FEATURES', fontsize=10)
    plt.title('FEATURES OF ALL THE FRAMES', fontsize=10)
    
    #plt.bar((19+i), len(kp), color='g')
    plt.bar((i), len(kp), color='g')

    plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
    #plt.title('FRAME-' + str(19+i), fontsize=10)
    plt.title('FRAME-' + str(i), fontsize=10)

    plt.imshow(newimg)

    plt.subplot(223)
    plt.xlabel('X-Coordinates', fontsize=10)
    plt.ylabel('Y-Coordinates', fontsize=10)
    plt.title('CAMERA POSE', fontsize=10)
    plt.xlim(left=-250)
    plt.xlim(right=1200)
    plt.ylim(-500, 1000)

    a1 = plt.scatter(df['X'][i], df['Y'][i], color='b',  marker='.', label='preefined-function')
    a2 = plt.scatter(df2['X'][i], df2['Y'][i], color='r', marker='.', label='our_code')
    plt.legend((a1, a2), ("Pre-def", "Code"), loc='center', fontsize=8)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.pause(0.0000001)

    plt.savefig('D:\\MD\\2nd sem\\perception\\pro5\\Oxford_dataset\\Oxford_dataset\\plots_final_ours\\'+str(i)+'.png')
    # plt.savefig('D:\\MD\\2nd sem\\perception\\pro5\\Oxford_dataset\\Oxford_dataset\\plots_final_rag\\'+str(i)+'.png')

#plt.show()

