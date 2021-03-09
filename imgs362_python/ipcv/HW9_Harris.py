if __name__ == '__main__':

    import os.path
    import time
    import cv2
    import ipcv
    import numpy

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
    filename = home + os.path.sep + 'src/python/examples/data/gecko.jpg'
    filename = home + os.path.sep + \
               'src/python/examples/data/sparse_checkerboard.tif'

    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    src = ipcv.map_rotation_scale()

    sigma = 1
    k = 0.01
    startTime = time.time()
    dst = ipcv.harris(src, sigma, k)
    print('Elapsed time = {0} [s]'.format(time.time() - startTime))

    cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename, src)

    if len(src.shape) == 2:
        annotatedImage = cv2.merge((src, src, src))
    else:
        annotatedImage = src
    fractionMaxResponse = 0.25
    annotatedImage[dst > fractionMaxResponse * dst.max()] = [0, 0, 255]

    cv2.namedWindow(filename + ' (Harris Corners)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(filename + ' (Harris Corners)', annotatedImage)

    print('Corner coordinates ...')
    indices = numpy.where(dst > fractionMaxResponse * dst.max())
    numberCorners = len(indices[0])
    if numberCorners > 0:
        for corner in range(numberCorners):
            print('({0},{1})'.format(indices[0][corner], indices[1][corner]))

    action = ipcv.flush()