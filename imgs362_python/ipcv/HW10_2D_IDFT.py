if __name__ == '__main__':
         import ipcv
         import numpy
         import time

         M = 2**5
         N = 2**5
         F = numpy.zeros((M,N), dtype=numpy.complex128)
         F[0,0] = 1

         repeats = 10
         print('Repetitions = {0}'.format(repeats))

         startTime = time.clock()
         for repeat in range(repeats):
            f = ipcv.idft2(F, verbose=True)
         string = 'Average time per transform = {0:.8f} [s] '
         string += '({1}x{2}-point iDFT2)'
         print(string.format((time.clock() - startTime)/repeats, M, N))

         startTime = time.clock()
         for repeat in range(repeats):
            f = numpy.fft.ifft2(F)
         string = 'Average time per transform = {0:.8f} [s] '
         string += '({1}x{2}-point iFFT2)'
         print(string.format((time.clock() - startTime)/repeats, M, N))
