#!/usr/bin/env python
# coding: utf-8

# In[1]:


##############################################################################
#######################Project1_AR_Homography#################################
#################### Part-3-Placing a Virtual Cube ###########################
# Team Members (Group Name - Project Group)
# PRANALI DESAI - 116182935
# NIKHIL MEHRA - 116189941
# SAYAN BRAHMA - 116309165
##############################################################################

import numpy as np
import cv2 as cv
import copy

##############################################################################
# Function Name - checkCorners
# Arguments - thresh (thresholded(binary) image), coord (coordinates to be checked)
# Returns - bool (True if the coordinate is the corner of the AR-Tag and False 
# otherwise), string ('TL' if the corner is a top-left corner, 'TR' if the corner
# is a top-right corner, 'BL' if the corner is a bottom-left corner, 'BR' if the
# corner is bottom right)
# Algorithm - It checks top left, top right, bottom left and bottom right areas
# corresponding to the given coordinate in the thresholded image. Through different
# combinations of these areas the function identifies the position of the corner 
# in the frame.
##############################################################################
def checkCorners(thresh, coord):
    
    count = 0 # Variable to check the number of white areas within the vicinity of the coordinate
    tl, tr, bl, br = 0, 0, 0, 0
    if coord[0] < 1900 and coord[1] < 1000: # To reject corners at the edge of the frame
        if thresh[coord[1]-10][coord[0]-10] == 255: # If the top left area corresponding to the coordinate is white
            count = count + 1
            tl = 1 # Then the variable 'tl(top left)' is assigned the value 1
        if thresh[coord[1]+10][coord[0]-10] == 255:# If the bottom left area corresponding to the coordinate is white
            count = count + 1
            bl = 1 # Then the variable 'bl(bottom left)' is assigned the value 1
        if thresh[coord[1]-10][coord[0]+10] == 255: # If the top right area corresponding to the coordinate is white
            count = count + 1
            tr = 1 # Then the variable 'tr(top right)' is assigned the value 1
        if thresh[coord[1]+10][coord[0]+10] == 255: # If the bottom right area corresponding to the coordinate is white
            count = count + 1
            br = 1 # Then the variable 'br(bottom right)' is assigned the value 1
            
        if count == 3: # If any three areas are white
            if tl == 1 and tr == 1 and bl == 1: # If top left, top right and bottom left are white
                return True, 'TL' # Then the function returns true and the string 'TL'
            elif tl == 1 and tr == 1 and br == 1: # If top left, top right and bottom right are white
                return True, 'TR' # Then the function returns true and the string 'TR'
            elif tl == 1 and br == 1 and bl == 1: # If top left, bottom right and bottom left are white
                return True, 'BL' # Then the function returns true and the string 'BL'
            elif tr == 1 and br == 1 and bl == 1: # If top right, bottom right and bottom left are white
                return True, 'BR' # Then the function returns true and the string 'BR'
        else: # Else if any three areas are not white
            return False, None # Then the function returns false and None
    else: # Else if the coordinate is within the edge of the frame
        return False, None # Then the function returns false and None
    
##############################################################################
# Function Name - homography
# Arguments - corners1 (coordinates of camera frame), corners2 (coordinates of world frame)
# Returns - H_matrix (Homography Matrix) 
# Algorithm - Calculates the 8 DOF of homography using SVD(Singular Value Decomposition)
##############################################################################
def homography(corners1, corners2):
    
    index = 0
    A = np.empty((8, 9))
    for i in range(0, len(corners1)):
        
        x1 = corners1[i][0] # Extracting x-coordinates of the camera frame element-wise
        y1 = corners1[i][1] # Extracting y-coordinates of the camera frame element-wise
        x2 = corners2[i][0] # Extracting x-coordinates of the world frame element-wise
        y2 = corners2[i][1] # Extracting y-coordinates of the world frame element-wise
        A[index] = np.array([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A[index + 1] = np.array([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
        index = index + 2
    
    U, s, V = np.linalg.svd(A, full_matrices=True) # Taking SVD of the matrix
    V = (copy.deepcopy(V)) / (copy.deepcopy(V[8][8])) # Extracting the eigen-vector corresponding to the least eigen-value
    H_matrix = V[8,:].reshape(3, 3) # Reshaping the eigen-vector into 3x3 format
    return H_matrix # Returns the 3x3 homography matrix

##############################################################################
# Function Name - projectionMatrix
# Arguments - h (homography matrix), calib (camera calibration matrix)
# Returns - P (Projection Matrix) 
# Algorithm - Calculates Projection matrix with respect to homography matrix,
# camera calibration matrix and scaling factor
##############################################################################
def projectionMatrix(h, calib):  # h is the homographic matrix and k is the camera calibration matrix
    h1 = h[:,0] # Column 1 of homographic matrix
    h2 = h[:,1] # Column 2 of homographic matrix
    h3 = h[:,2] # Column 3 of homographic matrix
    # Calculating scaling factor
    lam = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(calib),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(calib),h2)))
    b_tilda = lam * np.matmul(np.linalg.inv(calib),h)

    det = np.linalg.det(b_tilda) # Calculating determinant
    if det > 0: # If determinant is positive 
        b = b_tilda # Then the matrix is fine
    else: # Else if the determinant is negative
        b = -1 * b_tilda # Whole matrix is multiplied by -1

    r1 = b[:, 0] # Column 1 of b matrix
    r2 = b[:, 1] # Column 2 of b matrix
    r3 = np.cross(r1, r2) # Calculating cross product of Column 1 and Column 2 of b matrix
    t = b[:, 2] # Column 3 of b matrix
    R = np.column_stack((r1, r2, r3, t)) # Stacking r1,r2,r3 and t together to form a 3x4 matrix
    P = np.matmul(calib,R)  # Multiplying calibration matrix with R matrix
    return P # Returning Projection Matrix

##############################################################################
# Function Name - makeTagMatrix
# Arguments - img (image to be changed into 8x8 binary matrix)
# Returns - artag (8x8 binary matrix) 
# Algorithm - According to the height and width of the image, the image is 
# divided in 8x8 segments. Maximum operation is then applied, which finds whether
# there are greater number of white or black in the segment. The result of the 
# maximum operation is assigned to the corresponding segment.
##############################################################################
def makeTagMatrix(img):
    
    height = img.shape[0] # Height of the image
    width = img.shape[1] # Width of the image
    bitheight = int((height/8)) # Height of one segment
    bitwidth = int(width/8) # Width of one segment
    m=0
    n=0
    
    artag = np.empty((8,8))
    for i in range(0,height,bitheight): # Looping through the matrix by segment height
        n=0
        for j in range(0,width,bitwidth): # Looping through the matrix by segemnt width
            countblack = 0 # Variable for counting number of black pixels
            countwhite = 0 # Variable for counting number of white pixels
            for x in range(0,bitheight-1): # Looping through the each segment by height
                for y in range(0,bitwidth-1): # Looping through the each segment by width
                    if(img[i+x][j+y] == 0): # If the pixel value is 0 i.e. black
                        countblack = countblack + 1 # Increase the value of countblack by 1
                    else: # Else if the pixel value is 255 i.e. white
                        countwhite = countwhite + 1 # Increase the value of countwhite by 1
                        
            if(countwhite >= countblack): # If the number of white pixels are greater than number of black pixels
                artag[m][n]=1 # Then the correposding segment is assigned with value 1 i.e. white
            else: # Else if number of black pixels are greater than number of white pixels
                artag[m][n]=0 # Then the corresponding segment is assigned with the value of 0 i.e. black
            n=n+1
        m=m+1
    return artag # Returns the 8x8 binary matrix

##############################################################################
# Function Name - calculateTagAngle
# Arguments - tag (8x8 binary matrix)
# Returns - result (orientation of the AR Tag), bool (True if the orientation is
# identifiable, False otherwise) 
# Algorithm - Checks each corner of the AR tag after removing padding. Checks for
# a single white region among the four corners.
##############################################################################
def calculateTagAngle(tag):
    
    if(tag[2][2] == 0 and tag[2][5] == 0 and tag[5][2] == 0 and tag[5][5] == 1): # # If (5x5) element is white
        result = 0 # Then the orientation of the tag is 0 degree i.e. Original Configuration
    elif(tag[2][2] == 1 and tag[2][5] == 0 and tag[5][2] == 0 and tag[5][5] == 0): # If (2x2) element is white
        result = 180 # Then the orientation of the yag is 180 degrees i.e. Rotated upside down
    elif(tag[2][2] == 0 and tag[2][5] == 1 and tag[5][2] == 0 and tag[5][5] == 0): # If (2x5) element is white
        result = 90 # Then the orientation of the tag is 90 degrees 
    elif(tag[2][2] == 0 and tag[2][5] == 0 and tag[5][2] == 1 and tag[5][5] == 0): # If (5x2) element is white
        result = -90 # Then the orientation of the tag is -90 degrees 
    else:
        result = None # No orientation can be determined
        
    if (result == None): # If no orientation is determined
        return result, False # Then returns result as None and False
    else: # Else if the orientation is determined
        return result, True # Then returns orientation as result and True 
    
##############################################################################
# Function Name - calculateTagId
# Arguments - image (thresholded(binary) image)
# Returns - result (orientation of the AR Tag), bool (True if the orientation is
# identifiable, False otherwise) 
# Algorithm - Checks each corner of the AR tag after removing padding. Checks for
# a single white region among the four corners.
############################################################################## 
def calculateTagId(image):

    tagmatrix = makeTagMatrix(image) # Calling makeTagMatrix
    angle , flag = calculateTagAngle(tagmatrix) # Calling calculateTagAngle   
    if (flag == False): # If false is returned by calculateTagAngle i.e. angle was not identifiable     
        return flag , angle , None  # Returns False, None and None as nothing can be determined accurately
        
    if(flag == True): # If true is returned by calculateTagAngle i.e. angle is identifiable 
        if (angle == 0): # If the orientation is 0 degree
            identity = tagmatrix[3][3] +tagmatrix[4][3]*8 +tagmatrix[4][4]*4 + tagmatrix[3][4]*2 # Calculation of the tag id corresponding to the 0 degree configuration
        elif(angle == 90): # If the orientation is 90 degree
            identity = tagmatrix[3][3]*2 + tagmatrix[3][4]*4 + tagmatrix[4][4]*8 + tagmatrix[4][3] # Calculation of the tag id corresponding to the 90 degree configuration
        elif(angle == 180): # If the orientation is 180 degree
            identity = tagmatrix[3][3]*4 + tagmatrix[4][3]*2 + tagmatrix[4][4] + tagmatrix[3][4]*8 # Calculation of the tag id corresponding to the 180 degree configuration
        elif(angle == -90): # If the orientation is -90 degree
            identity = tagmatrix[3][3]*1 + tagmatrix[3][4] + tagmatrix[4][4]*2 + tagmatrix[4][3]*4 # Calculation of the tag id corresponding to the -90 degree configuration
        return flag, angle, identity # Returns True, orientation and tag id

vidname1 = 'multipleTags.mp4' # Input video 1
vidname2 = 'Tag0.mp4' # Input video 2
vidname3 = 'Tag1.mp4' # Input video 3
vidname4 = 'Tag2.mp4' # Input video 4

# Camera calibration matrix
calibration_matrix = np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]]).T

cap = cv.VideoCapture(vidname2) # Reading the video file

while (True): # Looping through all the frames
    
    ret, frame = cap.read()
    
    if not ret: # If no frame is generated or the video has ended
        cv.destroyAllWindows() # Destroy all Windows
        cap.release() # Releases software/hardware resource
        break

    blurred = cv.GaussianBlur(frame,(7,7),0) # Using a Gaussian Blur filter of size 7x7
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY) # Converting blurred frame into gray frame 
    ret, threshold = cv.threshold(gray, 240, 255, cv.THRESH_BINARY) # Thresholding the grayed frame
    _ , contours, _ = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # Finding contours in the thresholded frame
    list2 = [] # Variable for storing all necessary corners in the frame
    
    for contour in contours: # Looping over all the contours identified in the frame
        list1 = [] # Variable for storing corners with their keys for a particular contour
        if cv.contourArea(contour) > 1500: # If the area of contour is greater than 1500
            epsilon = 0.1*cv.arcLength(contour,True) # Approximating the contour using arcLength method
            approx = cv.approxPolyDP(contour, epsilon, True) # Determining the coordinates of the contour
            
            for coor in approx: # Looping over the corners found form the contour
                # Calling checkCorners function to check if the corner lies on AR Tag or not
                result, key = checkCorners(threshold, [coor[0][0], coor[0][1]]) 
                if result == True: # If the checkCorner function returns True
                    list1.append([coor[0][0], coor[0][1], key]) # Then the coordinates are appended into list1
        
            if len(list1) == 4: # If the length of list1 is 4 i.e. four corners on the AR tag are identified
                list2.append(list1) # Then the corners are appended to list2
      
    if list2 != []: # If list2 is not empty
        for i in range(0,len(list2)): # Looping over all the groups of corners identified
            list4 = [0,0,0,0] # Variable for storing the corners in a specified format i.e. ['TL','TR','BL','BR']
            for value in list2[i]: # Looping over the elements of a particular group of corners
                if value[-1] == 'TL': #  If the key is 'Tl' i.e. Top Left
                    list4[0] = value[0:2] # Then that element is assigned as the first element of the list4
                elif value[-1] == 'TR': #  If the key is 'TR' i.e. Top Right
                    list4[1] = value[0:2] # Then that element is assigned as the second element of the list4
                elif value[-1] == 'BL': #  If the key is 'BL' i.e. Bottom Left
                    list4[2] = value[0:2] # Then that element is assigned as the third element of the list4
                elif value[-1] == 'BR': #  If the key is 'BR' i.e. Bottom Right
                    list4[3] = value[0:2] # Then that element is assigned as the fourth element of the list4
                    
            if not(0 in list4): # If no '0' is present in list4 i.e. all the zeroes elements are replaced
                # coordinates in list4 corresponds to camera frame
                # [[0,0],[0,199],[199,0],[199,199]] corresponds to world frame
                H = homography(list4, [[0,0],[0,199],[199,0],[199,199]]) # Calculating homography between the camera frame and world frame
                H1 = np.linalg.inv(H) # Taking inverse of homographic matrix
                im_out = np.zeros((200,200)) # Making a new variable for showing AR-Tags
                
                for m in range(0,200): # Looping over the height of world frame
                    for n in range(0,200): # Looping over the width of world frame
                        xx, yy, zz = np.matmul(H1,[m,n,1]) # Calculating camera frame coordinates with respect to world frame coordinates
                        if (int(yy/zz) < 1080 and int(yy/zz) > 0) and (int(xx/zz) < 1920 and int(xx/zz) > 0):
                            im_out[m][n] = threshold[int(yy/zz)][int(xx/zz)] # Creating a new image from the camera frame 
                flag, angle_value, identity = calculateTagId(im_out) # calling calculateTagId
                
                if flag: # If id is identifiable
                    if angle_value == 0: # If the orientation is 0 degree
                        list3 = list4 # Then the format of the corners is correct
                    elif angle_value == 90: # If the orientation is 90 degree
                        list3 = [list4[2], list4[0], list4[3], list4[1]] # Then the format of the corners is corrected according to the configuration
                    elif angle_value == -90: # If the orientation is -90 degree
                        list3 = [list4[1], list4[3], list4[0], list4[2]] # Then the format of the corners is corrected according to the configuration
                    elif angle_value == 180: # If the orientation is 180 degree
                        list3 = [list4[3], list4[2], list4[1], list4[0]] # Then the format of the corners is corrected according to the configuration
                        
                    H1 = homography([[0,0],[0,199],[199,0],[199,199]],list3) # Calculating homography between the world frame and camera frame 
                    P = projectionMatrix(H1,calibration_matrix) # Calculating Projection matrix
                    # Camera Frame Coordinates = Projection Matrix * World Frame Coordinates
                    x1,y1,z1 = np.matmul(P,[0,0,0,1]) # Calculating camera frame coordinates for [0,0,0] 
                    x2,y2,z2 = np.matmul(P,[0,199,0,1]) # Calculating camera frame coordinates for [0,199,0] 
                    x3,y3,z3 = np.matmul(P,[199,0,0,1]) # Calculating camera frame coordinates for [199,0,0] 
                    x4,y4,z4 = np.matmul(P,[199,199,0,1]) # Calculating camera frame coordinates for [199,199,0] 
                    x5,y5,z5 = np.matmul(P,[0,0,-199,1]) # Calculating camera frame coordinates for [0,0,-199] 
                    x6,y6,z6 = np.matmul(P,[0,199,-199,1]) # Calculating camera frame coordinates for [0,199,-199] 
                    x7,y7,z7 = np.matmul(P,[199,0,-199,1]) # Calculating camera frame coordinates for [199,0,-199] 
                    x8,y8,z8 = np.matmul(P,[199,199,-199,1]) # Calculating camera frame coordinates for [199,199,-199] 
        
                    cv.circle(frame,(int(x1/z1),int(y1/z1)), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [0,0,0]
                    cv.circle(frame,(int(x2/z2),int(y2/z2)), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [0,199,0]
                    cv.circle(frame,(int(x3/z3),int(y3/z3)), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [199,0,0]
                    cv.circle(frame,(int(x4/z4),int(y4/z4)), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [199,199,0] 
                    cv.circle(frame,(int(x5/z5),int(y5/z5)), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [0,0,-199]
                    cv.circle(frame,(int(x6/z6),int(y6/z6)), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [0,199,-199]
                    cv.circle(frame,(int(x7/z7),int(y7/z7)), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [199,0,-199]
                    cv.circle(frame,(int(x8/z8),int(y8/z8)), 5, (0,0,255), -1) # Drawing a circle at the pixel w.r.t [199,199,-199]
                    # Drawing a line between pixels corresponding to [0,0,0] and [0,0,-199]
                    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (0,0,255), 5) 
                    # Drawing a line between pixels corresponding to [0,199,0] and [0,199,-199]
                    cv.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [199,0,0] and [199,0,-199]
                    cv.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [199,199,0] and [199,199,-199]
                    cv.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [0,0,0] and [0,199,0]
                    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [0,0,0] and [199,0,0]
                    cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [0,199,0] and [199,199,0]
                    cv.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [199,0,0] and [199,199,0]
                    cv.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [0,0,-199] and [0,199,-199]
                    cv.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [0,0,-199] and [199,0,-199]
                    cv.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [0,199,-199] and [199,199,-199]
                    cv.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (0,0,255), 5)
                    # Drawing a line between pixels corresponding to [199,0,-199] and [199,199,-199]
                    cv.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,0,255), 5)
 
    #break
    cv.imshow("Augmented_Reality_Virtual_Cube", frame) # Showing a new window with Virtual Cube placed over the AR-Tags
    if cv.waitKey(1) == 27: # Press 'ESC' to stop the processing and break out of the loop 
        cv.destroyAllWindows() # Destroys all window after pressing 'ESC'
        cap.release() # Releases software/hardware resource
        


# In[ ]:




