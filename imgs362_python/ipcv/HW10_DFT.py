if __name__ == '__main__':
    import matplotlib.pyplot as plot
    import ipcv
    import numpy
    import math
    import time
    import cmath
    N = 2 ** 8
    f = numpy.ones(N, dtype=numpy.complex128)

    f = (numpy.linspace(0,256,256))
    for x in range(0,len(f)):
       f[x] = (1/64)*(x**2)* math.sin(x)  # f = numpy.hstack((numpy.zeros(64), numpy.ones(128), numpy.zeros(64)))
    repeats = 1
    print('Repetitions = {0}'.format(repeats))

    startTime = time.clock()
    for repeat in range(repeats):
        F = ipcv.dft(f, scale = False, shift = False, verbose = True)
        f = ipcv.idft(F, scale = False, shift = False, verbose = True)
    string = 'Average time per transform = {0:.8f} [s] ({1}-point DFT)'
    print(string.format((time.clock() - startTime) / repeats, len(f)))

    startTime = time.clock()
    for repeat in range(repeats):
        F = numpy.fft.fft(f)
        plot.plot(F.real, 'r-')
        plot.plot(F.imag, 'b-')
        plot.show()

        recompiled_f = numpy.fft.ifft(F)

        plot.plot(recompiled_f.real, 'r-')
        plot.plot(recompiled_f.imag, 'b-')
        plot.show()
    string = 'Average time per transform = {0:.8f} [s] ({1}-point FFT)'
    print(string.format((time.clock() - startTime) / repeats, len(f)))
