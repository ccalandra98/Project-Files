if __name__ == '__main__':
         import numpy
         import cv2
         import ipcv
         import os.path
         import time

         home = os.path.expanduser('~')
         #filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
         filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
         #filename = home + os.path.sep + 'src/python/examples/data/photo1.png'
         #filename = home + os.path.sep + 'src/python/examples/data/jernigan_joseph_paul.jpg'

         src = cv2.imread(filename)
         startTime = time.clock()
         map1, map2 = ipcv.map_rotation_scale(src, rotation=45, scale=[2,2])
         elapsedTime = time.clock() - startTime
         print('Elapsed time (RST) = {0} [s]'.format(elapsedTime))

         startTime = time.clock()
         dst = ipcv.remap(src, map1, map2, cv2.INTER_NEAREST, borderMode = ipcv.BORDER_REPLICATE)
         elapsedTime = time.clock() - startTime
         print('Elapsed time (remap) = {0} [s]'.format(elapsedTime))

         srcName = 'Source (' + filename + ')'
         cv2.namedWindow(srcName, cv2.WINDOW_AUTOSIZE)
         cv2.imshow(srcName, src)

         dstName = 'Destination (' + filename + ')'
         cv2.namedWindow(dstName, cv2.WINDOW_AUTOSIZE)
         cv2.imshow(dstName, dst)

         ipcv.flush()