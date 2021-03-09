#   PYTHON TEST HARNESS
if __name__ == '__main__':
    import cv2
    import ipcv
    import os.path

    home = os.path.expanduser('~')
    jernigan = home + os.path.sep + 'src/python/examples/data/jernigan_joseph_paul.jpg'

    im = cv2.imread(jernigan, cv2.IMREAD_UNCHANGED)
    print('Filename = {0}'.format(jernigan))
    print('Data type = {0}'.format(type(im)))
    print('Image shape = {0}'.format(im.shape))
    print('Image size = {0}'.format(im.size))

    cv2.namedWindow(jernigan, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(jernigan, im)

    numberLevels = 7
    quantizedImage = ipcv.quantize(im,
                                   numberLevels,
                                   qtype='uniform',
                                   displayLevels=256)
    cv2.namedWindow(jernigan + ' (Uniform Quantization)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow(jernigan + ' (Uniform Quantization)', quantizedImage)
    action = ipcv.flush()

    directory = "D:\IPCV1"
    os.chdir(directory)

    #cv2.imwrite('GradientIGS_8_levels.tif', quantizedImage)


