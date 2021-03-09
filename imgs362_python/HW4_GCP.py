if __name__ == '__main__':

    import cv2
    import ipcv
    import os.path
    import time

    home = os.path.expanduser('~')
    imgFilename = home + os.path.sep + \
                  'src/python/examples/data/registration/image.tif'
    mapFilename = home + os.path.sep + \
                  'src/python/examples/data/registration/map.tif'
    gcpFilename = home + os.path.sep + \
                  'src/python/examples/data/registration/gcp.dat'
    src = cv2.imread(imgFilename)
    map = cv2.imread(mapFilename)

    srcX = []
    srcY = []
    mapX = []
    mapY = []
    linesRead = 0
    f = open(gcpFilename, 'r')
    for line in f:
        linesRead += 1
        if linesRead > 2:
            data = line.rstrip().split()
            srcX.append(float(data[0]))
            srcY.append(float(data[1]))
            mapX.append(float(data[2]))
            mapY.append(float(data[3]))
    f.close()

    startTime = time.clock()
    map1, map2 = ipcv.map_gcp(src, map, srcX, srcY, mapX, mapY, order=2)
    elapsedTime = time.clock() - startTime
    print('Elapsed time (map creation) = {0} [s]'.format(elapsedTime))

    startTime = time.clock()
    # dst = cv2.remap(src, map1, map2, cv2.INTER_NEAREST)
    dst = ipcv.remap(src, map1, map2, ipcv.INTER_NEAREST)
    elapsedTime = time.clock() - startTime
    print('Elapsed time (remap) = {0} [s]'.format(elapsedTime))

    srcName = 'Source (' + imgFilename + ')'
    cv2.namedWindow(srcName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(srcName, src)

    mapName = 'Map (' + mapFilename + ')'
    cv2.namedWindow(mapName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(mapName, map)

    dstName = 'Warped (' + mapFilename + ')'
    cv2.namedWindow(dstName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(dstName, dst)

    overlayName = 'Overlay (' + mapFilename + ')'
    cv2.namedWindow(overlayName, cv2.WINDOW_AUTOSIZE)
    overlay = cv2.addWeighted(map, 0.5, dst, 0.5, 0.0, dtype=-1)
    cv2.imshow(overlayName, overlay)

    ipcv.flush()
