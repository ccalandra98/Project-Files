import numpy
import math
import cv2
from numpy.linalg import eig as eigenvalue
def gaus2D(u, v, sigma):
    value = math.exp(1)**((-1/2)*((u**2+v**2)/(sigma**2)))
    return value

def harris(src, sigma=1, k=0.04):

    if len(src.shape) == 3:
        #Image is multiband, convert to monochrome single-valued array
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    srcshape = src.shape
    # initialize some variables and matrices

    #gaussian filtering window
    diameter = int(2 * (3/sigma) + 1)
    window = numpy.zeros((diameter,diameter))
    grid = numpy.indices((window.shape))-int(3/sigma)
    weights = numpy.zeros((window.shape))
    for i in range(0,window.shape[0]):
        for j in range(0,window.shape[1]):
            weights[i,j] = gaus2D(grid[0,i,j], grid[1,i,j], sigma)


    # Derivative "Images"
    partial = numpy.reshape(numpy.array([-1, 0, 1]),(3,1))
    X = cv2.filter2D((numpy.transpose(cv2.filter2D(numpy.transpose(src), cv2.CV_64F, partial))), cv2.CV_64F, weights)
    Y = cv2.filter2D((cv2.filter2D(src, cv2.CV_64F, partial)), cv2.CV_64F, weights)

    # Create A, B, and C for entire image
    A_Image = cv2.filter2D((X**2), cv2.CV_64F, weights)
    B_Image = cv2.filter2D((Y**2), cv2.CV_64F, weights)
    C_Image = cv2.filter2D((X*Y), cv2.CV_64F, weights)

    #Now calculate responses
    response = numpy.zeros(srcshape)
    for i in range(0,src.shape[0]):
        for j in range(0,src.shape[1]):
            A = A_Image[i,j]
            B = B_Image[i,j]
            C = C_Image[i,j]

            tr = A+B
            det = (A*B)-(C**2)

            response[i,j] = det-(k*tr**2)
    return response