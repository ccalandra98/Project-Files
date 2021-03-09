if __name__ == '__main__':
         import numpy
         import cv2
         import ipcv
         import os.path
         import time

         home = os.path.expanduser('~')
         imgFilename = home + os.path.sep + 'src/python/examples/data/Some Wack Shit.tif'
         mapFilename = home + os.path.sep + 'src/python/examples/data/gecko.jpg'
         img = cv2.imread(imgFilename)
         map = cv2.imread(mapFilename)

         mapName = 'Select corners for the target area (CW)'
         cv2.namedWindow(mapName, cv2.WINDOW_AUTOSIZE)
         cv2.imshow(mapName, map)

         print('')
         print('--------------------------------------------------------------')
         print('  Select the corners for the target area of the source image')
         print('  in clockwise order beginning in the upper left hand corner')
         print('--------------------------------------------------------------')
         p = ipcv.PointsSelected(mapName, verbose=True)
         while p.number() < 4:
            cv2.waitKey(100)
         cv2.destroyWindow(mapName)

         imgX = [0, img.shape[1]-1, img.shape[1]-1, 0]
         imgY = [0, 0, img.shape[0]-1, img.shape[0]-1]
         mapX = p.x()
         mapY = p.y()
         #mapX = (71, 29, 341, 387)
         #mapY = (91, 412, 450, 144)

         print('')
         print('Image coordinates ...')
         print('   x -> {0}'.format(imgX))
         print('   y -> {0}'.format(imgY))
         print('Target (map) coordinates ...')
         print('   u -> {0}'.format(mapX))
         print('   v -> {0}'.format(mapY))
         print('')

         startTime = time.clock()
         map1, map2 = ipcv.map_quad_to_quad(img, map, imgX, imgY, mapX, mapY)
         elapsedTime = time.clock() - startTime
         print('Elapsed time (map creation) = {0} [s]'.format(elapsedTime))

         startTime = time.clock()
         dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
         elapsedTime = time.clock() - startTime
         print('Elapsed time (remap) = {0} [s]'.format(elapsedTime))
         print('')

         compositedImage = map
         mask = numpy.where(dst != 0)
         if len(mask) > 0:
            compositedImage[mask] = dst[mask]

         compositedName = 'Composited Image'
         cv2.namedWindow(compositedName, cv2.WINDOW_AUTOSIZE)
         cv2.imshow(compositedName, compositedImage)

         ipcv.flush()
