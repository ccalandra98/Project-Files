if __name__ == '__main__':

         import cv2
         import os.path
         import time
         import numpy
         import ipcv

         home = os.path.expanduser('~')
         #filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
         #filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
         filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
         #filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
         filename = home + os.path.sep + 'src/python/examples/data/000044290016.jpg'

         src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

         dstDepth = ipcv.IPCV_8U
         #kernel = numpy.asarray([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
         #offset = 0
         #kernel = numpy.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
         #offset = 128
         kernel = numpy.ones((15,15))
         offset = 0
         #kernel = numpy.asarray([[1,1,1],[1,1,1],[1,1,1]])
         #offset = 0
         #kernel = numpy.asarray([[0, 2, 0], [2, 4, 2], [0, 2, 0]])
         #offset = 0
         #kernel = numpy.asarray([[5, 4, -9], [6, 5, -4], [-7, -8, 9]])
         #offset = 0
         #kernel = numpy.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
         #offset = 0

         startTime = time.time()
         dst = ipcv.filter2D(src, dstDepth, kernel, delta=offset)
         print('Elapsed time = {0} [s]'.format(time.time() - startTime))

         cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
         cv2.imshow(filename, src)

         cv2.namedWindow(filename + ' (Filtered)', cv2.WINDOW_AUTOSIZE)
         cv2.imshow(filename + ' (Filtered)', dst)

         action = ipcv.flush()