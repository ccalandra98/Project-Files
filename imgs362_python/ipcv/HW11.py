if __name__ == '__main__':

    import cv2
    import ipcv
    import os.path
    import sys
    import numpy
    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
    #filename = home + os.path.sep + 'src/python/examples/data/character_recognition/notAntiAliased/characters/81.tif'

    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #map1, map2 = ipcv.map_rotation_scale(im, 0, [2,2])
    #im = ipcv.remap(im,map1, map2)
    #im = numpy.reshape(im,(im.shape[0], im.shape[1]))
    if im is None:
        print('ERROR: Specified file did not contain a valid image type.')
        sys.exit(1)

#    ipcv.fft_display(im)
    ipcv.fft_display(im, videoFilename=None)