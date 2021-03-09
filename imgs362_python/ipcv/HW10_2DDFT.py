if __name__ == '__main__':
    import os.path
    import time
    import cv2
    import ipcv
    import numpy

    home = os.path.expanduser('~')
    filename = home + os.path.sep + 'src/python/examples/data/sparse_checkerboard.tif'
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    maps = ipcv.map_rotation_scale(src, 0 ,[0.5,0.5])
    src = ipcv.remap(src, maps[0], maps[1])

    M = (2 ** 8)
    N = (2 ** 8)
    f = numpy.ones((M, N), dtype=numpy.complex128)

    f = src.astype(numpy.complex128)
    if f.shape[2] == 1:
        f = numpy.reshape(f, (f.shape[0], f.shape[1]))
    #reals = f.real
    #imag = f.imag
    repeats = 2
    print('Repetitions = {0}'.format(repeats))

    startTime = time.clock()
    for repeat in range(repeats):
        F = ipcv.dft2(f, method = 'break', verbose = True)
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point DFT2)'
    print(string.format((time.clock() - startTime) / repeats, M, N))

    startTime = time.clock()
    for repeat in range(repeats):
        F = numpy.fft.fft2(f)
        reals = F.real
        imag = F.imag
    string = 'Average time per transform = {0:.8f} [s] '
    string += '({1}x{2}-point FFT2)'
    print(string.format((time.clock() - startTime) / repeats, M, N))