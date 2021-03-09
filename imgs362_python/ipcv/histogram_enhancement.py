def histogram_enhancement(im, etype='linear2', target=None, maxCount=255, showHistogram=False, userInputs=False):
    import numpy
    import matplotlib.pyplot as plot

    # Extra arguments showHistogram == True: program will display histogram/CDF from original and modified image
    # userInputs == True: allows you to input user-specified values for certain modification types, such as the
    # cutoff range for the linear2 histogram modification, and whether to do rolled color channels versus individual
    # color channels for histogram matching.
    shape_im = im.shape
    shape_target = target.shape
    histogramFlag = 'rolled'
    if len(
            shape_im) == 2:  # determines if the given image is a 2D greyscale array or 3D array. If 2D, converts to 3D greyscale array
        shape_im_3D = (shape_im[0], shape_im[1], 3)  # for ease of calculation.
        im3D = numpy.zeros(shape_im_3D)
        for n in range(0, 3):
            im3D[:, :, n] = im
            n = n + 1
        im = im3D
    else:
        n = 0
    if len(
            shape_target) == 2:  # determines if the given image is a 2D greyscale array or 3D array. If 2D, converts to 3D greyscale array
        shape_target_3D = (shape_target[0], shape_target[1], 3)  # for ease of calculation.
        target3D = numpy.zeros(shape_target_3D)
        for n in range(0, 3):
            target3D[:, :, n] = target
            n = n + 1
        target = target3D
    else:
        n = 0
    im = im.astype(int)
    # compute original image histograms
    num_bins = maxCount + 1
    counts, bin_edges = numpy.histogram(im, bins=num_bins, range=(0, maxCount), density=False)
    im_pdf = counts / im.size
    im_cdf = numpy.cumsum(counts) / im.size

    if etype == 'linear1':
        # compute rise/run to find slope, where rise = desired range (0,255) and run is current range (DCmin,DCmax).
        # then use slope to find y intercept and come up with a linear LUT
        rise = maxCount
        run = int(numpy.max(im)) - int(numpy.min(im))
        slope = rise / run
        b = 0 - (slope * int(numpy.min(im)))
        LUT = numpy.linspace(0, maxCount, maxCount + 1) * (
            slope) + b  # building the base LUT in the form of LUT = slope(0:255)+b
        LUT = LUT.astype(int)
        n = 0
        for n in range(0, maxCount + 1):  # clipping function for the LUT
            if LUT[n] >= maxCount:
                LUT[n] = maxCount
            elif LUT[n] <= 0:
                LUT[n] = 0
            n = n + 1
        output = numpy.take(LUT, im)  # numpy.take is a very fast LUT applicator.
        output = output.astype(
            numpy.uint8)  # outputting to UINT8 for display. This would need to be changed if we were outputting to a different bit depth.
    elif etype == 'linear2':
        # We need to find the cutoff values. Best way to do this is to subtract the cutoff percentiles (I will do 5 and 95)
        # from the CDF LUT, take the absolute value, and then take the minimum.
        if userInputs == True:  # userInputs flag allows one to specify the boundary if desired.
            trimamount = float(input("Specify the percentage for boundary cutoff (for example, 5% --> 0.05): "))
        else:
            trimamount = 0.02  # The amount being trimmed off either histogram.
        input_lo = trimamount
        input_hi = 1 - trimamount

        CDF_locut = abs(im_cdf - input_lo)
        CDF_hicut = abs(im_cdf - input_hi)
        CDF_locut = numpy.ndarray.tolist(CDF_locut)
        CDF_hicut = numpy.ndarray.tolist(CDF_hicut)

        pos_locut = CDF_locut.index(min(CDF_locut))
        pos_hicut = CDF_hicut.index(min(CDF_hicut))

        rise = maxCount
        run = pos_hicut - pos_locut  # Run is the index of the lowcut and highcut CVs
        # from here on out, same as linear1
        slope = rise / run
        b = 0 - (slope * int(numpy.min(im)))
        LUT = numpy.linspace(0, maxCount, maxCount + 1) * (slope) + b
        LUT = LUT.astype(int)
        n = 0
        for n in range(0, maxCount + 1):
            if LUT[n] >= maxCount:
                LUT[n] = maxCount
            elif LUT[n] <= 0:
                LUT[n] = 0
            n = n + 1

        output = numpy.take(LUT, im)
        output = output.astype(numpy.uint8)

    elif etype == 'equalize':
        # Start by dividing out maximum bit depth to scale between 0 and 1
        LUT = (numpy.linspace(0, 1, maxCount + 1) * im_cdf) * maxCount  # scale by the CDF, then return to 0-255 scale
        LUT = LUT.astype(numpy.uint8)
        output = numpy.take(LUT, im)
        output = output.astype(numpy.uint8)

    elif etype == 'match':
        # create finding function for array indexing
        import numpy as np
        def find_nearest(array, value):  # I found this function on stackoverflow, linking here for transparency:
            array = np.asarray(array)  # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
            idx = (np.abs(
                array - value)).argmin()  # The function takes an array, and searches for the closest index to the value given.
            return array[idx]

        if userInputs == True:  # This specifies whether you can do independent channel matching or "rolled together" matching with one histogram.
            # Defaults to rolled together.
            histogramFlag = input(
                "Specify 'rolled' for rolled histogram matching, or 'independent' for channel-independent histogram matching:")

        if histogramFlag == 'rolled':
            # Check flag first
            if len(target.shape) >= 1:
                # This conditional is passing through an image to match histograms if the number of dimensions is greater than 1.
                # Match probability from source CDF to target CDF, take the index value there. This becomes the lookup CV at the source CV at matching probability.
                source_cdf = im_cdf  # Create the original source PDF and CDF from image
                num_bins = maxCount + 1
                target_counts, bin_edges = numpy.histogram(target, bins=num_bins, range=(0, maxCount), density=False)
                bin_edges = numpy.linspace(0, maxCount, maxCount + 1)
                target_pdf = target_counts / target.size
                target_cdf = numpy.cumsum(target_pdf)
                LUT = numpy.zeros(maxCount + 1)
                nearest_target = numpy.zeros(maxCount + 1)
                n = 0
                # Use the probability desired to match to in the find_nearest function, then store this in
                # the "nearest_targets" array to find indices. Indices form the LUT.
                for n in range(0, maxCount + 1):
                    matching_probability = source_cdf[n]
                    nearest_target[n] = find_nearest(target_cdf, matching_probability)
                    n = n + 1
                target_cdf = numpy.ndarray.tolist(target_cdf)
                for n in range(0, maxCount + 1):
                    LUT[n] = target_cdf.index(nearest_target[n])
                    n = n + 1

            else:
                source_cdf = im_cdf  # Same procedure as before, but program detects that the matching image is a pre-built
                target_pdf = target  # LUT by determining the size of the array beforehand.
                target_cdf = numpy.cumsum(target_pdf)
                LUT = numpy.zeros(maxCount + 1)
                nearest_target = numpy.zeros(maxCount + 1)
                n = 0
                for n in range(0, maxCount + 1):
                    matching_probability = source_cdf[n]
                    nearest_target[n] = find_nearest(target_cdf, matching_probability)
                    n = n + 1
                target_cdf = numpy.ndarray.tolist(target_cdf)  # Converting to list such that I can index from the list
                for n in range(0, maxCount + 1):
                    LUT[n] = target_cdf.index(nearest_target[n])
                    n = n + 1
        elif histogramFlag == 'independent':  # Independent channel matching is an experiment more for myself than anything.
            if len(target.shape) >= 1:  # It has the same procedure as rolled matching, but simply computes for three
                num_bins = maxCount + 1  # independent color bands, so additional loops were nested.
                n = 0
                counts = numpy.zeros((3, maxCount + 1))
                target_counts = numpy.zeros((3, maxCount + 1))
                im_pdf = numpy.zeros((3, maxCount + 1))
                target_pdf = numpy.zeros((3, maxCount + 1))
                source_cdf = numpy.zeros((3, maxCount + 1))
                target_cdf = numpy.zeros((3, maxCount + 1))
                LUT = numpy.zeros((3, maxCount + 1))
                nearest_target = numpy.zeros((3, maxCount + 1))
                for n in range(0, 3):
                    counts[n], bin_edges = numpy.histogram(im[:, :, n], bins=num_bins, range=(0, maxCount),
                                                           density=False)
                    im_pdf[n] = counts[n] / (im.size / 3)
                    source_cdf[n] = numpy.cumsum(counts[n]) / (im.size / 3)
                    n = n + 1
                for n in range(0, 3):
                    target_counts[n], bin_edges = numpy.histogram(target[:, :, n], bins=num_bins, range=(0, maxCount),
                                                                  density=False)
                    target_pdf[n] = target_counts[n] / (target.size / 3)
                    target_cdf[n] = numpy.cumsum(target_pdf[n])
                    n = n + 1

                for n in range(0, 3):
                    cdf_list = target_cdf
                    for m in range(0, maxCount + 1):
                        matching_probability = source_cdf[n, m]
                        nearest_target[n, m] = find_nearest(target_cdf[n, :], matching_probability)
                        m = m + 1
                    cdf_list = numpy.ndarray.tolist(cdf_list[n, :])
                    for m in range(0, maxCount + 1):
                        LUT[n, m] = cdf_list.index(nearest_target[n, m])
                        m = m + 1
                    n = n + 1
            else:
                print("Error: Must pass in image for per-channel matching to work")
                exit()
        LUT = LUT.astype(numpy.uint8)
        output = numpy.take(LUT, im)
        output = output.astype(numpy.uint8)

    if showHistogram == True:  # this is an optional conditional that will permit you to view the histograms before and
        if histogramFlag == 'independent':  # after enhancement if desired. Defaults to off.
            print("Still working on independent channel histograms!")
            return output
        # create output histograms for reference
        num_bins = maxCount + 1
        output_counts, bin_edges = numpy.histogram(output, bins=num_bins, range=(0, maxCount), density=False)
        bin_edges = numpy.linspace(0, maxCount, maxCount + 1)
        output_pdf = output_counts / output.size
        output_cdf = numpy.cumsum(output_counts) / output.size

        fig, axs = plot.subplots(4, 1, sharex=True)
        plot.suptitle('Distributions Before and After Enhancement', horizontalalignment='center',
                      verticalalignment='top')
        fig.subplots_adjust(hspace=0.25)

        axs[0].plot(bin_edges[0:], im_pdf, '-b', label='Plot of Original PDF')
        axs[0].set_xlim(0, 255)

        axs[1].plot(bin_edges[0:], im_cdf, '-r', label='Plot of Original CDF')
        axs[1].set_xlim(0, 255)
        axs[1].set_ylim(0, 1)

        axs[2].plot(bin_edges[0:], output_pdf, '-y', label='PDF enhanced via: {etype}')
        axs[2].set_xlim(0, 255)

        axs[3].plot(bin_edges[0:], output_cdf, '-g', label='CDF enhanced via: {etype}')
        axs[3].set_xlim(0, 255)
        axs[3].set_ylim(0, 1)
        axs[3].set_xlabel('Digital Count')

        fig.legend(loc='center right', fontsize='x-small')
        plot.show()

    return output
