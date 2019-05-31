# #######################Project3_GMM_Fitting################################
# Team Members (Group Name - Project Group)
# PRANALI DESAI - 116182935
# NIKHIL MEHRA - 116189941
# SAYAN BRAHMA - 116309165
##############################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import copy


def calculateGaussianEquation(xcoor, mean, std):
    return ((1 / (std * math.sqrt(2 * math.pi))) * np.exp(-np.power(xcoor - mean, 2.) / (2 * np.power(std, 2.))))


def prepareHistogramData(imagePath, numberOfImages, channels, size=(24, 24)):
    endimage = np.zeros(size)

    for i in range(1, numberOfImages + 1):
        image = cv.imread(str(imagePath) + str(i) + ".jpg")  # "./red_buoy/" +str(i) +".jpg"
        shape = image.shape

        if shape[0] >= size[0] and shape[1] >= size[1]:
            image1 = np.zeros((shape[0], shape[1]))

            for channel in channels:
                image1 = image1 + image[:, :, channel]

            image1 = image1 / (len(channels))

            endimage = endimage + image1[int((shape[0] / 2) - size[0] / 2):int((shape[0] / 2) + size[0] / 2),
                                  int((shape[1] / 2) - size[1] / 2):int((shape[1] / 2) + size[1] / 2)]

    endimage = endimage / numberOfImages
    l1 = np.zeros((1, 256))

    for j in range(0, endimage.shape[0]):
        for k in range(0, endimage.shape[1]):
            l1[0][int(endimage[j, k])] = l1[0][int(endimage[j, k])] + 1

    return np.squeeze(l1)


redbuoy_r = prepareHistogramData("./red_buoy/", 131, [2])
plt.plot(range(0, 256), redbuoy_r, 'r')
plt.show()

meanr = sum(redbuoy_r * range(0, 256)) / sum(redbuoy_r)
stdr = (np.sum(redbuoy_r * (np.array(range(0, 256)) - meanr) ** 2) / sum(redbuoy_r)) ** (1 / 2)
print(meanr, stdr)

plt.plot(range(0, 256), calculateGaussianEquation(redbuoy_r, meanr, stdr), 'r')
plt.xlim([0, 255])
plt.show()
'''
y1 = np.squeeze(l1)
y2 = np.squeeze(l2)
y3 = np.squeeze(l3)
plt.plot(range(0,256), y1, 'b')
plt.plot(range(0,256), y2, 'g')
plt.plot(range(0,256), y3, 'r')
plt.show()

meanb = 0
meang = 0
meanr = 0
for i in range(0, len(y1)):
    meanb = meanb + i*y1[i]
    meang = meang + i*y2[i]
    meanr = meanr + i*y3[i]

meanb = meanb/(1.0*sum(y1))
meang = meang/(1.0*sum(y2))
meanr = meanr/(1.0*sum(y3))
print(meanb, meang, meanr)

stdb = (np.sum((np.array(range(0,256)) - meanb)**2)/256)**(1/2)
stdg = (np.sum((np.array(range(0,256)) - meang)**2)/256)**(1/2)
stdr = (np.sum((np.array(range(0,256)) - meanr)**2)/256)**(1/2)
print(stdb, stdg, stdr)

plt.plot(range(0,256), gaussian(y1 ,meanb, stdb), 'b')
plt.show()
plt.plot(range(0,256), gaussian(y2 ,meang, stdg), 'g')
plt.show()
plt.plot(range(0,256), gaussian(y3 ,meanr, stdr), 'r')
plt.show()
'''