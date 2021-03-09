import numpy
import math
import cv2
import ipcv

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
