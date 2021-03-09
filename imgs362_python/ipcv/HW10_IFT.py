if __name__ == '__main__':
    import ipcv
    import numpy
    import time

    N = 2 ** 8
    F = numpy.zeros(N, dtype=numpy.complex128)
    F[0] = 1

    repeats = 10
    print('Repetitions = {0}'.format(repeats))

    startTime = time.clock()
    for repeat in range(repeats):
        f = ipcv.idft(F, verbose = True, scale = False)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point iDFT)'
    print(string.format((time.clock() - startTime) / repeats, len(F)))

    startTime = time.clock()
    for repeat in range(repeats):
        f = numpy.fft.ifft(F)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point iFFT)'
    print(string.format((time.clock() - startTime) / repeats, len(F)))