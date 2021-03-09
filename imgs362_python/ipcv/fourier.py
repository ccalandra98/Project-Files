import numpy
import math
import cv2
import matplotlib.pyplot as plot
import ipcv

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
def dft2(f, scale=True, method = 'break', verbose = False):

    #initialize variables
    shape = f.shape
    j = complex(0, 1)
    e = math.exp(1)
    M = shape[0]
    N = shape[1]
    pi = math.pi
    f_x_y = numpy.zeros((shape), dtype=numpy.complex128)
    dst = numpy.zeros(shape, dtype = numpy.complex128)

    # checkerboard image to center
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            f[x,y] = f[x,y]*pow(-1,(x+y))

    #now compute fourier transform
    if method == 'together': #discrete, four-loop method
        for u in range(0,shape[0]):
            for v in range(0,shape[1]):
                for x in range(0,shape[0]):
                    for y in range(0,shape[1]):
                        f_x_y[x,y] = f[x,y]*(e**(-j*2*pi*(((u*x)/M)+((v*y)/N))))
                if scale == True:
                    dst[u,v] = sum(sum(f_x_y))/(M*N)
                else:
                    dst[u,v] = sum(sum(f_x_y))
    if method == 'break': #modified, less computationally-heavy method
        f_x_v = numpy.zeros(shape, dtype = numpy.complex128)
        for x in range(0,M): #use DFT algorithm twice, computing rows first, then columns.
            f_x_v[x,:] = dft(f[x,:],scale = scale, shift = False, verbose = False)

        for y in range(0,N):
            dst[:,y] = dft(f_x_v[:,y], scale = scale, shift = False, verbose = False)

    # Optional display method
    if verbose == True:
        f_complex = dst[:, :] + 1j * dst[:, :]
        f_abs = numpy.abs(f_complex) + 1  # lie between 1 and 1e6
        f_bounded = 20 * numpy.log(f_abs)
        f_img = 255 * f_bounded / numpy.max(f_bounded)
        f_img = f_img.astype(numpy.uint8)
        cv2.imshow('F(z) of Source Image',f_img)
        ipcv.flush()
        return dst
    else:
        return dst


def idft2(F, scale=True, method='break', verbose=False):
    # initialize variables
    shape = F.shape
    j = complex(0, 1)
    e = math.exp(1)
    M = shape[0]
    N = shape[1]
    pi = math.pi
    f_x_y = numpy.zeros((shape), dtype=numpy.complex128)
    dst = numpy.zeros(shape, dtype=numpy.complex128)

    # checkerboard image to center
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            F[x, y] = F[x, y] * pow(-1, (x + y))

    # now compute fourier transform
    if method == 'together':  # discrete, four-loop method
        for u in range(0, shape[0]):
            for v in range(0, shape[1]):
                for x in range(0, shape[0]):
                    for y in range(0, shape[1]):
                        f_x_y[x, y] = F[x, y] * (e ** (-j * 2 * pi * (((u * x) / M) + ((v * y) / N))))
                if scale == True:
                    dst[u, v] = sum(sum(f_x_y)) / (M * N)
                else:
                    dst[u, v] = sum(sum(f_x_y))
    if method == 'break':  # modified, less computationally-heavy method
        f_x_v = numpy.zeros(shape, dtype=numpy.complex128)
        for x in range(0, M):  # use DFT algorithm twice, computing rows first, then columns.
            f_x_v[x, :] = dft(F[x, :], scale=scale, shift=False, verbose=False)

        for y in range(0, N):
            dst[:, y] = dft(f_x_v[:, y], scale=scale, shift=False, verbose=False)

    # Optional display method
    if verbose == True:
        f_complex = dst[:, :] + 1j * dst[:, :]
        f_abs = numpy.abs(f_complex) + 1  # lie between 1 and 1e6
        f_bounded = 20 * numpy.log(f_abs)
        f_img = 255 * f_bounded / numpy.max(f_bounded)
        f_img = f_img.astype(numpy.uint8)
        cv2.imshow('F(x) of Source Image', f_img)
        ipcv.flush()
        return dst
    else:
        return dst
