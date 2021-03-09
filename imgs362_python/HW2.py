if __name__ == '__main__':

         import cv2
         import ipcv
         import os.path
         import time
         import numpy

         home = os.path.expanduser('~')
         #filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
         #filename = home + os.path.sep + 'src/python/examples/data/Split Tree.tif'
         #filename = home + os.path.sep + 'src/python/examples/data/Niagara River Infrared +2EV.tif'
         #filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
         filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
         #filename = home + os.path.sep + 'src/python/examples/data/Linear.tif
         #filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
         #filename = home + os.path.sep + 'src/python/examples/data/photo1.png'
         #filename = home + os.path.sep + 'src/python/examples/data/photo2.png'
         #filename = home + os.path.sep + 'src/python/examples/data/Cherries.tiff'
         #filename = home + os.path.sep + 'src/python/examples/data/000044290016.jpg'
         filename = home + os.path.sep + 'src/python/examples/data/Some Wack Shit.tif'

         #matchFilename = home + os.path.sep + 'src/python/examples/data/Cherries.tiff'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/giza.jpg'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/Linear.tif'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/Accordion Man.jpg'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/Niagara River Infrared +2EV.tif'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/MessinAround.tif'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/photo1.png'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/gecko.jpg'
         #matchFilename = home + os.path.sep + 'src/python/examples/data/000044290016.jpg'
         matchFilename = home + os.path.sep + 'src/python/examples/data/Some Wack Shit.tif'

         im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
         print('Filename = {0}'.format(filename))
         print('Data type = {0}'.format(type(im)))
         print('Image shape = {0}'.format(im.shape))
         print('Image size = {0}'.format(im.size))

         cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
         cv2.imshow(filename, im)

         tgtIm = cv2.imread(matchFilename, cv2.IMREAD_UNCHANGED)
         tgtPDF = numpy.ones(256) / 256
         print('Matched (Distribution) ...')
         startTime = time.time()
         enhancedImage = ipcv.histogram_enhancement(im, etype='equalize', target=tgtPDF, userInputs = False, showHistogram = True)
         print('Elapsed time = {0} [s]'.format(time.time() - startTime))
         cv2.namedWindow(filename + ' (Matched - Distribution)', cv2.WINDOW_AUTOSIZE)
         cv2.imshow(filename + ' (Matched - Distribution)', enhancedImage)
         cv2.imwrite('testVillage.tif',enhancedImage)

         action = ipcv.flush()
