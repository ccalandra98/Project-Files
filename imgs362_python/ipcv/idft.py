import numpy
import math
import matplotlib.pyplot as plot

def idft(F, scale=True, shift = False, verbose = False):
    # Initialize constants
    j = complex(0, 1)
    e = math.exp(1)
    M = len(F)
    pi = math.pi
    F_z = numpy.zeros((len(F)), dtype=numpy.complex128)
    dst = numpy.zeros((len(F)), dtype=numpy.complex128)

    #optional shift to center array
    if shift == True:
        #shift the array such that the fourier transform is centered.
        for x in range(0,F.size):
            F[x] = F[x]*pow(-1,x)
    #implement fourier transform.
    for u in range(0, F.size):
        for x in range(0, len(F)):  # both methods work, first method is slightly faster per computation (tested at M = 256)
            F_z[x] = F[x] * (e ** ((j * 2 * pi * u * x) / M))
            #F_z[x] = F[x] * (cos(2 * pi * u * x / M) - (j * sin(2 * pi * u * x / M)))

        #optional scaling
        if scale == True:
            dst[u] = sum(F_z) / M
        else:
            dst[u] = sum(F_z)
    #optional display
    if verbose == True:
        plot.plot(numpy.linspace(0, F.size - 1, F.size), dst.real, 'r-')
        plot.plot(numpy.linspace(0, F.size - 1, F.size), dst.imag, 'b-')
        plot.show()
    return dst

