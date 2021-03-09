import ipcv
import numpy
from numpy.linalg import norm
from numpy import matmul
from numpy import dot as ndot
from numpy import sum as nsum
from numpy import reshape
import cv2
import math
import time
e = math.exp(1)


def find_center(a):  # Function used to find array center. Used for indexing purposes. Didn't make myself. Credit here:
    # https://stackoverflow.com/questions/54084527/clean-way-of-returning-middle-index-of-matrix-in-python user: DYZ
    x = (a.shape[0] // 2 if a.shape[0] % 2 else a.shape[0] // 2 - 1,
         a.shape[0] // 2 + 1)
    y = (a.shape[1] // 2 if a.shape[1] % 2 else a.shape[1] // 2 - 1,
         a.shape[1] // 2 + 1)
    center = a[x[0]:x[1], y[0]:y[1]]
    argmax = numpy.unravel_index(center.argmax(), center.shape)
    return argmax[0] + x[0], argmax[1] + y[0]  # Adjust


def bilateral_filter(src, sigmaDistance, sigmaRange, d=-1, borderType=ipcv.BORDER_WRAP, borderVal=0, maxCount=255):
    orig_shape = src.shape
    orig_size = src.size
    if len(orig_shape) == 2:
        # convert to mxnx1 array for ease of calculation
        src = numpy.reshape(src, (orig_shape[0], orig_shape[1], 1))

    dst = numpy.zeros(src.shape)
    # d = filter radius. If negative, must equal double the sigma d value. Use this value to adjust the array borders too.
    if d < 0:
        d = 2 * sigmaDistance
    else:
        pass
    d = int(d)
    # work on border modes
    npad = ((d, d), (d, d), (0, 0))
    if borderType == ipcv.BORDER_WRAP:
        src = numpy.pad(src, npad, mode='wrap')
    elif borderType == ipcv.BORDER_CONSTANT:
        src = numpy.pad(src, npad, mode='constant', constant_values=borderVal)
    elif borderType == ipcv.BORDER_REFLECT:
        src = numpy.pad(src, npad, mode='reflect')
    else:
        print("Border mode not supported. Please use 'BORDER_WRAP', 'BORDER_CONSTANT',"
              " or 'BORDER_REFLECT'.")
        exit()

    if len(orig_shape) == 3:
        # This is a color image. Perform CIELAB calculation to convert color space.

        if src.dtype == numpy.uint8:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        else:
            print("Error: Must input 8-bit-image.")
            exit()

    elif len(orig_shape) == 2:
        # This is a greyscale image. Values will represent luminance. Pass through
        pass
    else:
        print("Error: source image passed is neither a color or greyscale image.\n 'src' should be either a 3D 3-channel\
        color image or a 2D greyscale image.")

    # Now that our arrays are padded appropriately, we can start the filtering process.

    closeness = numpy.zeros((1 + (2 * d), 1 + (2 * d))) # Definitions for the size/shape of filter.
    similarity = numpy.zeros((1 + (2 * d), 1 + (2 * d)))
    bilateralfilter = numpy.zeros((1 + (2 * d), 1 + (2 * d)))

    center = numpy.array(find_center(bilateralfilter))

    # These two loops are the initiators for the entire image's filter. We now iterate pixel-by-pixel to calculate the
    # bilateral filter.

    #Create the closeness filter once to prevent re-iteration.
    for i in range(0, bilateralfilter.shape[0]):
        for j in range(0, bilateralfilter.shape[1]):
            distance = numpy.array((i, j))
            closeness[i, j] = e ** (-.5 * ((norm(center - distance) / sigmaDistance) ** 2))

    count = 0
    if len(orig_shape) == 2:
        for columns in range(d, orig_shape[0] + d):
            startTime = time.time()
            for rows in range(d, orig_shape[1] + d):
                # These loops initiate the iterative process for creating the bilateral filter.
                for i in range(0, bilateralfilter.shape[0]):
                    for j in range(0, bilateralfilter.shape[1]):
                        similarity[i, j] = e ** (-.5 * ((abs(
                            src[columns, rows, 0] - src[columns + (i - d), rows + (j - d), 0]) / sigmaRange) ** 2))
                bilateralfilter = matmul(closeness, similarity)
                bilateralfilter = bilateralfilter / nsum(nsum(bilateralfilter))
                srcrange = src[columns - d:columns + d + 1, rows - d:rows + d + 1]
                dst[columns - d, rows - d] = ndot(reshape(srcrange, (-1)), reshape(bilateralfilter, (-1)))
                count = count + 1
           # print('Row Completion Time: = {0} [s]'.format(time.time() - startTime))
           # print("Percentage Complete: ", 100 * (count / orig_size))

    # For Color Images
    else:
        luminance = numpy.zeros(dst[:,:,0].shape)
        for columns in range(d, orig_shape[0] + d):
            startTime = time.time()
            for rows in range(d, orig_shape[1] + d):
                # These loops initiate the iterative process for creating the bilateral filter.
                for i in range(0, bilateralfilter.shape[0]):
                    for j in range(0, bilateralfilter.shape[1]):
                        similarity[i, j] = e ** (-.5 * ((abs(src[columns, rows, 0] - src[columns + (i - d), rows + (j - d), 0]) / sigmaRange) ** 2))
                bilateralfilter = matmul(closeness, similarity)
                bilateralfilter = bilateralfilter / nsum(nsum(bilateralfilter))
                srcrange = src[columns - d:columns + d + 1, rows - d:rows + d + 1, 0]
                luminance[columns - d, rows - d] = ndot(reshape(srcrange,(-1)), reshape(bilateralfilter,(-1)))
                count = count + 1
            #print('Row Completion Time: = {0} [s]'.format(time.time() - startTime))
            #print("Percentage Complete: \n {0}%".format(100 * (count*3 / orig_size)))
            #print("")



    # dst is created. Now we need to return to original image state. If 2D, simply is quantized and clipped. If 3D,
    # convert back to RGB colorspace.

    if len(orig_shape) == 2:
        dst = dst.astype(int)
        dst = numpy.reshape(numpy.clip(dst, 0, maxCount).astype(numpy.uint8), (orig_shape[0], orig_shape[1]))
    else:
        dst[: ,: ,0] = luminance
        dst[:, :, 1] = src[d: orig_shape[0] + d, d: orig_shape[1] + d, 1]
        dst[:, :, 2] = src[d: orig_shape[0] + d, d: orig_shape[1] + d, 2]
        dst = dst.astype(numpy.uint8)
        dst = cv2.cvtColor(dst, cv2.COLOR_LAB2RGB)
        dst = numpy.clip(dst, 0, maxCount).astype(numpy.uint8)
    return dst
