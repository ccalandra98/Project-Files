import cv2
import numpy
import math


from numpy.fft import ifft2, fftshift
def fft_display(im, videoFilename=None):

    # video filename must not contain file extension. will output a .mpeg.
    # must display 6 unique image display windows named "Original Image", "Fourier Transform - log(magnitude)",
    # "Fourier Coefficients Used - log(magnitude)", "Current Component", "Current Component (Scaled)", and "Summed
    # Components" that update as each new component is added. The "p" or "P" keys must toggle (pause/resume)
    # "animation", the "Esc", "q", or "Q" keys must terminate the program.

    imshape = im.shape
    im = im/255 #assuming uint8

    # take fourier transform
    im_fft = numpy.fft.fft2(im)

    #Get FT Magnitude
    fft_magnitudes = numpy.log10(numpy.sqrt((im_fft.real**2)+(im_fft.imag**2)))

    #get log magnitudes FOR DISPLAY
    f_complex = im_fft[:, :] + 1j * im_fft[:, :]
    f_abs = numpy.abs(f_complex) + 1  # lie between 1 and 1e6
    f_bounded = 20 * numpy.log(f_abs)
    f_img = 255 * f_bounded / numpy.max(f_bounded)
    f_img = f_img.astype(numpy.uint8)/255
    log10fft = numpy.roll(f_img,(int(f_img.shape[0]/2)), axis = 0)
    log10fft = numpy.roll(log10fft, (int(f_img.shape[1]/2)), axis = 1)


   #designate frame size, which will be a box of six images in 3x2 arrangement
    frameSize = tuple((int(im.shape[0] * 2), int(im.shape[1] * 3)))

    #if there is a filename, we write the video
    if videoFilename:
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        isColor = False
        videoShape = (frameSize[1], frameSize[0])  # Note that the VideoWriter
        # object frame size is
        # (columns, rows) while a
        # numpy array is (rows, columns)
        writer = cv2.VideoWriter(videoFilename, codec, fps, videoShape, isColor)

    #Array init
    recomposited_fft = numpy.array(numpy.zeros((im.shape))).astype(complex)
    current_sinusoid = numpy.array(numpy.zeros((im.shape))).astype(complex)
    display_fft = numpy.zeros((im.shape))

    #Variable inits
    numberFramesWritten = 0
    stepFlag = False
    count = 0
    try:
        while True:

            # We need to get the index of the greatest magnitudes in the fourier image, in order. we then take these
            # magnitudes, build the sinusoids from the index we pulled from, scale, and add to the image.

            workingindex =  numpy.unravel_index(fft_magnitudes.argmax(), imshape) #Get current maximum magnitude index

            # Then rebuild the fft map, take fourier transform to get recomposited dst. Each loop adds a frequency
            recomposited_fft[workingindex[0],workingindex[1]] = im_fft[workingindex[0],workingindex[1]]
            recomposited_dst = ifft2(recomposited_fft)

            current_sinusoid[workingindex[0],workingindex[1]] = im_fft[workingindex[0],workingindex[1]]

            display_sinusoid = ifft2(current_sinusoid).real
            display_sinusoid_scaled = (display_sinusoid-numpy.min(display_sinusoid))/numpy.ptp(display_sinusoid)
            real_additive_sinusoid = display_sinusoid+.5

            display_fft[workingindex[0],workingindex[1]] = log10fft[workingindex[0],workingindex[1]]

            # Build the frame
            upper_half_frame = numpy.hstack((im,log10fft,display_sinusoid_scaled))
            bottom_half_frame = numpy.hstack((recomposited_dst.real,display_fft,real_additive_sinusoid))
            frame = numpy.vstack((upper_half_frame,bottom_half_frame))


            #reset current sinusoid, increase count
            current_sinusoid = numpy.array(numpy.zeros((im.shape))).astype(complex)
            count = count +1



            if videoFilename:
                writer.write(frame)
                numberFramesWritten += 1

            cv2.imshow('recombined image', frame)
            fft_magnitudes[workingindex[0],workingindex[1]] = -1 # make sure these values won't be picked again
            print("Index test: ", workingindex)
            #create the pause/play buttons. Also build an individual step button by using "s" key
            if stepFlag == False:
                k = cv2.waitKey(10)
            if k == ord('p') or k == ord('P'):
                if stepFlag == False:
                    print('Paused: Press "p" again to continue')
                while k == ord('p'):
                    resume = cv2.waitKey(10)
                    if resume == ord('p') or resume == ord('P'):
                        print('Continuing...')
                        stepFlag = False
                        break
                    if resume == ord('s') or resume == ord('S'):
                        print('Stepping')
                        k = ord('p')
                        stepFlag = True
                        break
                    if resume == 27:
                        k = 27
                        break
            if count == im.size:
                print('Image reconstruction complete!')
                break


            if k == 27:
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print('')

    print('{0} frames written'.format(numberFramesWritten))
    return 0
