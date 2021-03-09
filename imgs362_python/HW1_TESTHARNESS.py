PYTHON TEST HARNESS
      if __name__ == '__main__':

         import cv2
         import ipcv
         import os.path

         home = os.path.expanduser('~')
         filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
         filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
         filename = home + os.path.sep + 'src/python/examples/data/linear.tif'
         filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'

         im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
         print('Filename = {0}'.format(filename))
         print('Data type = {0}'.format(type(im)))
         print('Image shape = {0}'.format(im.shape))
         print('Image size = {0}'.format(im.size))

         cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
         cv2.imshow(filename, im)

         numberLevels = 7
         quantizedImage = ipcv.quantize(im,
                                        numberLevels,
                                        qtype='uniform',
                                        displayLevels=256)
         cv2.namedWindow(filename + ' (Uniform Quantization)', cv2.WINDOW_AUTOSIZE)
         cv2.imshow(filename + ' (Uniform Quantization)', quantizedImage)

         numberLevels = 7
         quantizedImage = ipcv.quantize(im,
                                        numberLevels,
                                        qtype='igs',
                                        displayLevels=256)
         cv2.namedWindow(filename + ' (IGS Quantization)', cv2.WINDOW_AUTOSIZE)
         cv2.imshow(filename + ' (IGS Quantization)', quantizedImage)

         action = ipcv.flush()