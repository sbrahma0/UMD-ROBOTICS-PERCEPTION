import numpy as np
import matplotlib.pyplot as plt
import random  # Package used for randomize technique in RANSAC(Random Sample Consensus)
from sympy.solvers import solve  # For solving linear equations in order find eigenvalue and eigenvector
from sympy import Symbol  # Package used to define a variable so as to solve a linear equation

####################################################################
################Question1-Geometrical Interpretation################
####################################################################

data1 = np.array(np.load('data1_new.pkl')).T  # loading data from data1_new.pkl
data2 = np.array(np.load('data2_new.pkl')).T  # loading data from data1_new.pkl
data3 = np.array(np.load('data3_new.pkl')).T  # loading data from data1_new.pkl

x1 = data1[0]  # X-coordinates of data-1
y1 = data1[1]  # Y-coordinates of data-1
x2 = data2[0]  # X-coordinates of data-2
y2 = data2[1]  # Y-coordinates of data-2
x3 = data3[0]  # X-coordinates of data-3
y3 = data3[1]  # Y-coordinates of data-3


#####################################################################
# Function-Name - calculateVariance
# Input - Data in any one varaible (list type)
# Output - Variance of the given data (float type)
# Algoritm - (Summation of (x - xmean))/(Total Number of Points)
#####################################################################

def calculateVariance(data):
    l1 = data[:]  # Copy of the data
    size = len(l1)  # Size of the data
    val = ((l1 - np.mean(l1)) ** 2)
    return np.sum(val) / size


#####################################################################
# Function-Name - calculateCovariance
# Input - Data of two variables, two one dimensional array (list type)
# Output - Covariance of the given data (float type)
# Algoritm - (Summation of (x - xmean)*(y - ymean))/(Total Number of Points)
#####################################################################

def calculateCovariance(datax, datay):
    l1 = datax[:]  # Copy of the data
    l2 = datay[:]  # Copy of the data
    size = len(l1)  # Size of the data
    val = (l1 - np.mean(l1)) * (l2 - np.mean(l2))
    return np.sum(val) / size


#####################################################################
# Function-Name - calculate2DCovarianceMatrix
# Input - Data of two variables, two one dimensional array (list type)
# Output - Covariance Matrix of the given data (2 Dimensional Matrix)
# Algoritm - Calls calculateVariance and calculateCovariance Functions and assigns
# Covariances on the diagonal and variance on the non-diagonal term in a 2D array
#####################################################################

def calculate2DCovarianceMatrix(xcoor, ycoor):
    return [[calculateVariance(xcoor), calculateCovariance(xcoor, ycoor)],
            [calculateCovariance(xcoor, ycoor), calculateVariance(ycoor)]]


cov1 = calculate2DCovarianceMatrix(x1, y1)  # Covariance Matrix of data1
cov2 = calculate2DCovarianceMatrix(x2, y2)  # Covariance Matrix of data2
cov3 = calculate2DCovarianceMatrix(x3, y3)  # Covariance Matrix of data3

lamda = Symbol('lamda')


#####################################################################
# Function-Name - calculateEigen
# Input - 2x2 Matrix
# Output - Eigenvalues and Eigenvectors in a nested list where eigenvalues are
# at zero index and eigenvectors are at first index
# Algorithm - Solves the characteristics equation of the matrix and determines the
# eigenvalues. Solves simulatenous linear equation (Av=(lambda)v) to determine the eigenvectors.
#####################################################################

def calculateEigen(mat):
    if mat[0][0] == 1 and mat[0][1] == 0 and mat[1][0] == 0 and mat[1][1] == 1:  # If the matrix is an identity matrix
        return [[1, 1], [[1, 0], [0, 1]]]  # Then it returns eigenvalues as [1,1] and eigenvector as [[1,0],[0,1]]
    dt = ((mat[0][0] - lamda) * (mat[1][1] - lamda) - (mat[0][1] * mat[1][0]))  # Characteristic Equation of the matrix
    sol = solve(dt, lamda)  # Solution of the characteristic Equation using sympy package
    if len(sol) == 1:  # If the solution of the quadratic equation is a single value
        sol = [sol[0], sol[0]]  # Then the solution will be of repeated roots
    sol.sort()  # Sorting the eigenvalues in ascending order
    sol.reverse()  # Reversing the eigenvalues in order to get them in desending order
    eigvec = []
    for eig in sol:  # Loop for calculating eigenvector for each eigenvalue
        x = (1 / (1 + (mat[1][0] ** 2) / ((eig - mat[1][1]) ** 2))) ** (1 / 2)
        y = (mat[1][0] * x) / (eig - mat[1][1])
        eigvec.append([x, y])
    eigval = [float(sol[0]), float(sol[1])]
    return [eigval, list((np.array(eigvec).T))]


eig1, eigv1 = calculateEigen(cov1)  # Eigenvalues and Eigenvectors for corresponding covariance matrix of data1
eig2, eigv2 = calculateEigen(cov2)  # Eigenvalues and Eigenvectors for corresponding covariance matrix of data2
eig3, eigv3 = calculateEigen(cov3)  # Eigenvalues and Eigenvectors for corresponding covariance matrix of data3

plt.figure(1)  # Plot one is for presenting the Geometrical Interpretation of data1
plt.scatter(x1, y1)  # Scatter plot of data1
plt.quiver([0, float(eigv1[0][0])], [0, float(eigv1[1][0])], color='r', scale=5)  # Plotting major axis
plt.quiver([0, float(eigv1[0][1])], [0, float(eigv1[1][1])], scale=10)  # Plotting minor axis
# plt.savefig('Question1 - Data1') # Remove '#' before plt to save the figure
plt.xlabel('Question1 - Data1')

plt.figure(2)  # Plot one is for presenting the Geometrical Interpretation of data2
plt.scatter(x2, y2)  # Scatter plot of data2
plt.quiver([0, float(eigv2[0][0])], [0, float(eigv2[1][0])], color='r', scale=5)  # Plotting major axis
plt.quiver([0, float(eigv2[0][1])], [0, float(eigv2[1][1])], scale=10)  # Plotting minor axis
# plt.savefig('Question1 - Data2') # Remove '#' before plt to save the figure
plt.xlabel('Question1 - Data2')

plt.figure(3)  # Plot one is for presenting the Geometrical Interpretation of data3
plt.scatter(x3, y3)  # Scatter plot of data3
plt.quiver([0, float(eigv3[0][0])], [0, float(eigv3[1][0])], color='r', scale=5)  # Plotting major axis
plt.quiver([0, float(eigv3[0][1])], [0, float(eigv3[1][1])], scale=10)  # Plotting minor axis
# plt.savefig('Question1 - Data3') # Remove '#' before plt to save the figure
plt.xlabel('Question1 - Data3')


#######################################################################
##########Question2-Vertical and Orthogonal Line Fitting###############
#######################################################################

#######################################################################
# Funtion Name - lineFittingVerticalDistance
# Input - x-coordinates and y-coordinates of the given data
# Output - x-coordinate and y-coordinate of the line using vertical offset
# Algorithm - Minimizing the summation of the square of the vertical offset with
# respect to the line (y = mx + c). Hence solving the equation B = inv(transpose(X)*X)*(transpose(X)*Y)
#######################################################################

def lineFittingVerticalDistance(xcoor, ycoor):
    Y = ycoor[:]  # Copying the y-coordinates
    Y.shape = (len(xcoor), 1)
    X = np.vstack((xcoor, np.ones(
        (len(xcoor))))).T  # Converting 200x1 size x-cordinates into 200x2 matrix, where 2nd column consists only 1
    sol = np.matmul(np.linalg.inv(np.matmul(X.T, X)),
                    np.matmul(X.T, Y))  # Solving the equation B = inv(transpose(X)*X)*(transpose(X)*Y)
    # solution of the above equation gives slope and intercept
    x_vals = list(range(-100, 101))  # Assigning values to x-coordinate of the line
    y_vals = []
    for num in range(0, len(x_vals)):
        y_vals.append(sol[1][0] + sol[0][0] * x_vals[
            num])  # Calculating y-coordinate of the line depending upon the values of x-coordinates, slope and intercept
    return x_vals, y_vals


#######################################################################
# Funtion Name - lineFittingVerticalDistanceWithRegularization
# Input - x-coordinates and y-coordinates of the given data
# Output - x-coordinate and y-coordinate of the line using vertical offset
# Minimizing the summation of the square of the vertical offset with
# respect to the line (y = mx + c) with an extra regularization term (lambda*I). Where lambda are
# the eigen values and I is an identity matrix. Hence solving the equation
# B = inv(transpose(X)*X + (lambda*I))*(transpose(X)*Y)
#######################################################################

def lineFittingVerticalDistanceWithRegularization(xcoor, ycoor):
    Y = ycoor[:]  # Copying the y-coordinates
    Y.shape = (len(xcoor), 1)
    X = np.vstack((xcoor, np.ones(
        (len(xcoor))))).T  # Converting 200x1 size x-cordinates into 200x2 matrix, where 2nd column consists only 1
    covm = calculate2DCovarianceMatrix(xcoor, ycoor)  # Calculating the covariance matrix
    eig, eigv = calculateEigen(covm)  # Calculating the eigenvalues and eigenvectors of the covariance matrix
    sol = np.matmul(np.linalg.inv(np.matmul(X.T, X) + np.matmul(eig, [[1, 0], [0, 1]])),
                    np.matmul(X.T, Y))  # Solving the equation B = inv(transpose(X)*X +(lambda*I))*(transpose(X)*Y)
    x_vals = list(range(-100, 101))  # Assigning values to x-coordinate of the line
    y_vals = []
    for num in range(0, len(x_vals)):
        y_vals.append(sol[1][0] + sol[0][0] * x_vals[
            num])  # Calculating y-coordinate of the line depending upon the values of x-coordinates, slope and intercept
    return x_vals, y_vals


########################################################################
# Funtion Name - lineFittingOrthogonalDistance
# Input - x-coordinates and y-coordinates of the given data
# Output - x-coordinate and y-coordinate of the line using perpendicular offset
# Algorithm - Minimizing the summation of the square of the orthogonal offest with
# respect to the line (ax+by=d). Hence solving the equation (transpose(U)*U*N = 0)
########################################################################

def lineFittingOrthogonalDistance(xcoor, ycoor):
    xmean = np.mean(xcoor)  # Calculating mean of the x-coordinates
    ymean = np.mean(ycoor)  # Calculating mean of the y-coordinates
    X = xcoor - xmean  # Calculating (X-Xmean)
    Y = ycoor - ymean  # Calculating (Y-Ymean)
    U = np.vstack((X, Y)).T  # Stacking both (X-Xmean) and (Y-Ymean)
    UtransU = np.matmul(U.T, U)  # Multiplying (transpose(U)*U)
    eig, eigv = calculateEigen(UtransU)  # Calculating eigenvalues and eigenvectors of (transpose(U)*U)
    d = eigv[0][1] * xmean + eigv[1][1] * ymean  # Calculating d = a*xmean + b*mean
    x_vals = list(range(-100, 101))  # Assigning values to x-coordinate of the line
    y_vals = []
    for index in range(0, len(x_vals)):  # Loop for calculating y-coordinates of the line
        y_vals.append(((-eigv[0][1] * x_vals[index]) - d) / (eigv[1][1]))  # Depending upon the 'a' and 'b' values
    return x_vals, y_vals


lfox1, lfoy1 = lineFittingOrthogonalDistance(x1, y1)  # X and Y coordinates of the Orthogonal Fit line of data1
lfvrx1, lfvry1 = lineFittingVerticalDistanceWithRegularization(x1,
                                                               y1)  # X and Y coordinates of the Vertical Fit line with Regularization of data1
lfvx1, lfvy1 = lineFittingVerticalDistance(x1, y1)  # X and Y coordinates of the Vertical Fit line of data1

lfox2, lfoy2 = lineFittingOrthogonalDistance(x2, y2)  # X and Y coordinates of the Orthogonal Fit line of data2
lfvrx2, lfvry2 = lineFittingVerticalDistanceWithRegularization(x2,
                                                               y2)  # X and Y coordinates of the Vertical Fit line with Regularization of data2
lfvx2, lfvy2 = lineFittingVerticalDistance(x2, y2)  # X and Y coordinates of the Vertical Fit line of data2

lfox3, lfoy3 = lineFittingOrthogonalDistance(x3, y3)  # X and Y coordinates of the Orthogonal Fit line of data3
lfvrx3, lfvry3 = lineFittingVerticalDistanceWithRegularization(x3,
                                                               y3)  # X and Y coordinates of the Vertical Fit line with Regularization of data3
lfvx3, lfvy3 = lineFittingVerticalDistance(x3, y3)  # X and Y coordinates of the Vertical Fit line of data3

plt.figure(4)  # Plotting OLS, TLS and LS+Regularization for data1
plt.scatter(x1, y1)  # Scatter plot of data1
plt.plot(lfvx1, lfvy1, color='r')  # Plotting Least Square Fit with Vertical Offset
plt.plot(lfox1, lfoy1, color='k')  # Plotting Least Square Fit with Orthogonal Offset
plt.plot(lfvrx1, lfvry1, color='g')  # Plotting Least Square Fit with Vertical Offset with Regularization
plt.legend(('OLS', 'TLS', 'LS+Reg', 'Data'), loc='best')
plt.savefig('Question2 - Data1')  # Remove '#' before plt to save the figure
plt.xlabel('Question2 - Data1')

plt.figure(5)  # Plotting OLS, TLS and LS+Regularization for data2
plt.scatter(x2, y2)  # Scatter plot of data2
plt.plot(lfvx2, lfvy2, color='r')  # Plotting Least Square Fit with Vertical Offset
plt.plot(lfox2, lfoy2, color='k')  # Plotting Least Square Fit with Orthogonal Offset
plt.plot(lfvrx2, lfvry2, color='g')  # Plotting Least Square Fit with Vertical Offset with Regularization
plt.legend(('OLS', 'TLS', 'LS+Reg', 'Data'), loc='best')
plt.savefig('Question2 - Data2')  # Remove '#' before plt to save the figure
plt.xlabel('Question2 - Data2')

plt.figure(6)  # Plotting OLS, TLS and LS+Regularization for data3
plt.scatter(x3, y3)  # Scatter plot of data3
plt.plot(lfvx3, lfvy3, color='r')  # Plotting Least Square Fit with Vertical Offset
plt.plot(lfox3, lfoy3, color='k')  # Plotting Least Square Fit with Orthogonal Offset
plt.plot(lfvrx3, lfvry3, color='g')  # Plotting Least Square Fit with Vertical Offset with Regularization
plt.legend(('OLS', 'TLS', 'LS+Reg', 'Data'), loc='best')
plt.savefig('Question2 - Data3')  # Remove '#' before plt to save the figure
plt.xlabel('Question2 - Data3')


###################################################################
###################Question3-RANSAC################################
###################################################################

###################################################################
# Funtion Name - RANSAC
# Input - x-coordinates, y-coordinates of the given data, threshold distance and numOfInliers
# Output - x-coordinate and y-coordinate of the inliers i.e. Model after rejecting
# Algorithm - Its a random iterative technique used to find a filtered model of
# the data i.e. with reduced noise. Arguments threshold and numOfInliers can be
# tweaked to reach a better filtered version of the model.
###################################################################

def RANSAC(xcoor, ycoor, threshold, numOfInliers):
    inliers = []  # list for inliers
    outliers = []  # list for outliers
    flag = 0  # Variable to keep the while loop in check. Maximum Iteration could be 5000
    while (True):
        point1 = random.randint(0, 199)  # First random point
        point2 = random.randint(0, 199)  # Second random point
        for num in range(0, 200):
            distn = (ycoor[num] - ycoor[point1]) * (xcoor[point2] - xcoor[point1]) - (
                        (ycoor[point2] - ycoor[point1]) * (xcoor[num] - xcoor[point1]))  # Calculating abs(aXi+bYi+d)
            distd = ((xcoor[point2] - xcoor[point1]) ** 2 + (ycoor[point2] - ycoor[point1]) ** 2) ** (
                        1 / 2)  # Calculating (a**2 + b**2)**(1/2)
            value = distn / distd  # Calculating the distance of each point from the line
            if abs(value) <= threshold:  # If the distance is less than threshold
                inliers.append(num)  # Then the data point is considered as an inlier
            else:
                outliers.append(num)  # Else the data point is considered as an outlier
        if len(inliers) < numOfInliers:  # If the size of the inliers list is lesser than the numOfInlier argument
            inliers = []  # Then the random sample taken in this iteration does not satisfy our model
            outliers = []
        else:
            xnew = []  # Else if the size of inliers list is greater than the numOfInliers argument
            ynew = []  # Then list of X,Y coordinates are created for each inliers and outliers
            xold = []
            yold = []
            for index in inliers:
                xnew.append(xcoor[index])
                ynew.append(ycoor[index])
            for index in outliers:
                xold.append(xcoor[index])
                yold.append(ycoor[index])
            return np.array(xnew), np.array(ynew), np.array(xold), np.array(yold), np.array(
                [xcoor[point1], ycoor[point1]]), np.array([xcoor[point2], ycoor[point2]])
        if flag == 5000:  # If Iteration goes beyond 5000 then the loop breaks
            break
        else:
            flag = flag + 1


########################################################################
# Funtion Name - lineFittingOrthogonalDistance2
# Website - http://mathworld.wolfram.com/LeastSquaresFittingPerpendicularOffsets.html
# Input - x-coordinates and y-coordinates of the given data
# Output - x-coordinate and y-coordinate of the line using perpendicular offset
# Algorithm - Minimizing the summation of the square of the orthogonal offest with
# respect to the line (y = a + bx). Hence solving the equations
# B = ((sum(y**2)-n*ymean**2)-(sum(y**2)-n*ymean**2))/(2*(n*xmean*ymean - sum(x*y)))
# b = -B + (B**2 + 1) and b = -B - (B**2 + 1)
########################################################################

def lineFittingOrthogonalDistance2(xcoor, ycoor):
    xmean = np.mean(xcoor)  # Calculating mean of the x-coordinates
    ymean = np.mean(ycoor)  # Calculating mean of the y-coordinates
    size = len(xcoor)
    X = xcoor[:]
    Y = ycoor[:]
    # Solving the equation B = ((sum(y**2)-n*ymean**2)-(sum(y**2)-n*ymean**2))/(2*(n*xmean*ymean - sum(x*y)))
    B = ((np.sum(Y ** 2) - (size * (ymean) ** 2)) - (np.sum(X ** 2) - (size * (xmean) ** 2))) / (
                2 * ((size * xmean * ymean) - (np.sum(X * Y))))
    # Solvinf the equations b = -B + (B**2 + 1) and b = -B - (B**2 + 1)
    b1 = -B - (B ** 2 + 1) ** (1 / 2)  # 'b' for line1
    b2 = -B + (B ** 2 + 1) ** (1 / 2)  # 'b' for line2
    a1 = ymean - b1 * xmean  # 'a' for line1
    a2 = ymean - b2 * xmean  # 'a' for line2
    x_vals = list(range(-100, 101))  # Assigning values to x-coordinate of the line
    y_vals1 = []
    y_vals2 = []
    for index in range(0, len(x_vals)):  # Loop for calculating the y-coordinates of line1 and line2
        y_vals1.append(a1 + b1 * x_vals[index])  # Calculating the values of y-coordinates for line1
        y_vals2.append(a2 + b2 * x_vals[index])  # Calculating the values of y-coordinates for line2

    return x_vals, y_vals2


x1new, y1new, x1old, y1old, randpoint11, randpoint12 = RANSAC(x1, y1, 10,
                                                              170)  # Determining filtered model for data1 using threshold value as 10 and numofInliers as 170
x2new, y2new, x2old, y2old, randpoint21, randpoint22 = RANSAC(x2, y2, 10,
                                                              120)  # Determining filtered model for data2 using threshold value as 10 and numofInliers as 120
x3new, y3new, x3old, y3old, randpoint31, randpoint32 = RANSAC(x3, y3, 10,
                                                              70)  # Determining filtered model for data3 using threshold value as 10 and numofInliers as 70

lfox1new, lfoy1new = lineFittingOrthogonalDistance2(x1new,
                                                    y1new)  # Calculating X,Y coordiantes for Orthogonal Offset Technique on Filtered Model in data1
lfvx1new, lfvy1new = lineFittingVerticalDistance(x1new,
                                                 y1new)  # Calculating X,Y coordiantes for Vertical Offset Technique on Filtered Model in data1

lfox2new, lfoy2new = lineFittingOrthogonalDistance2(x2new,
                                                    y2new)  # Calculating X,Y coordiantes for Orthogonal Offset Technique on Filtered Model in data2
lfvx2new, lfvy2new = lineFittingVerticalDistance(x2new,
                                                 y2new)  # Calculating X,Y coordiantes for Vertical Offset Technique on Filtered Model in data2

lfox3new, lfoy3new = lineFittingOrthogonalDistance2(x3new,
                                                    y3new)  # Calculating X,Y coordiantes for Orthogonal Offset Technique on Filtered Model in data3
lfvx3new, lfvy3new = lineFittingVerticalDistance(x3new,
                                                 y3new)  # Calculating X,Y coordiantes for Vertical Offset Technique on Filtered Model in data3

plt.figure(7)  # Plotting RANSAC and TLS on the filtered model in data1
plt.scatter(x1new, y1new)  # Scatter plot of Inliers data1
plt.scatter(x1old, y1old, color='g')  # Scatter plot of Outliers data1
plt.plot([randpoint11[0], randpoint12[0]], [randpoint11[1], randpoint12[1]],
         color='r')  # Plotting RANSAC of the filtered model in data1
plt.plot(lfox1new, lfoy1new,
         color='k')  # Plotting least square fit using Orthogonal offset technique on the filtered model in data1
plt.legend(('RANSAC', 'RANSAC+TLS2', 'Inliers', 'Outliers'), loc='best')
plt.savefig('Question3 - Data1')  # Remove '#' before plt to save the figure
plt.xlabel('Question3 - Data1')

plt.figure(8)  # Plotting RANSAC and TLS on the filtered model in data2
plt.scatter(x2new, y2new)  # Scatter plot of Inliers data2
plt.scatter(x2old, y2old, color='g')  # Scatter plot of Outliers data3
plt.plot([randpoint21[0], randpoint22[0]], [randpoint21[1], randpoint22[1]],
         color='r')  # Plotting RANSAC of the filtered model in data2
plt.plot(lfox2new, lfoy2new,
         color='k')  # Plotting least square fit using Orthogonal offset technique on the filtered model in data2
plt.legend(('RANSAC', 'RANSAC+TLS2', 'Inliers', 'Outliers'), loc='best')
plt.savefig('Question3 - Data2')  # Remove '#' before plt to save the figure
plt.xlabel('Question3 - Data2')

plt.figure(9)  # Plotting RANSAC and TLS on the filtered model in data3
plt.scatter(x3new, y3new)  # Scatter plot of Inliers data3
plt.scatter(x3old, y3old, color='g')  # Scatter plot of Outliers data3
plt.plot([randpoint31[0], randpoint32[0]], [randpoint31[1], randpoint32[1]],
         color='r')  # Plotting RANSAC of the filtered model in data3
plt.plot(lfox3new, lfoy3new,
         color='k')  # Plotting least square fit using Orthogonal offset technique on the filtered model in data3
plt.legend(('RANSAC', 'RANSAC+TLS2', 'Inliers', 'Outliers'), loc='best')
plt.savefig('Question3 - Data3')  # Remove '#' before plt to save the figure
plt.xlabel('Question3 - Data3')

plt.show()

# In[ ]:




# In[ ]:
