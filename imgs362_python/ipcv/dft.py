import numpy
import math
import matplotlib.pyplot as plot

def dft(f, scale=True,shift = False, verbose = False):
    j = complex(0,1)
    e = math.exp(1)
    M = len(f)
    pi = math.pi
    f_x = numpy.zeros((len(f)),dtype=numpy.complex128)
    dst = numpy.zeros((len(f)),dtype=numpy.complex128)

    # optional shift to center array
    if shift == True:
        #shift the array such that the fourier transform is centered.
        for x in range(0,f.size):
            f[x] = f[x]*pow(-1,x)

    # implement fourier transform.
    for u in range(0,f.size):
        for x in range(0,len(f)): #both methods work, first method is slightly faster per computation (tested at M = 256)
            f_x[x] = f[x]*(e**((-j*2*pi*u*x)/M))
            #f_x[x] = f[x]*(cos(2*pi*u*x/M)+(j*sin(2*pi*u*x/M)))
        #optional scaling
        if scale == True:
            dst[u] = sum(f_x)/M
        else:
            dst[u] = sum(f_x)
    #optional display
    if verbose == True:
        plot.plot(numpy.linspace(0, f.size-1, f.size), dst.real, 'r-')
        plot.plot(numpy.linspace(0, f.size-1, f.size), dst.imag, 'b-')
        plot.show()
    return dst

