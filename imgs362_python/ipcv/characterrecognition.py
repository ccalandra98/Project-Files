def character_recognition(src, templates, codes, threshold, filterType='spatial'):
    import numpy
    import matplotlib
    import matplotlib.pyplot as plt

    orig_shape = src.shape

    # Matched Filtering Technique
    if filterType == 'matched':

        # start by normalizing source image to (0,1)
        if src.dtype and templates.dtype == numpy.uint8:
            src = (src / 255).astype(int)
            templates = (templates / 255).astype(int)
        else:
            print("One or more of these images are not of dtype: 8-bit integer")
            exit()

        # pad source with whitespace
        newsrc = numpy.zeros((orig_shape[0] + templates.shape[1], orig_shape[1] + templates.shape[2]))

        # invert the images for ease of later calculations
        src = abs(src-1).astype(float)
        templates = abs(templates-1).astype(float)

        #scale the templates such that sum = 1
        for i in range(0, templates.shape[0]):
            checksum = sum(sum(templates[i, :, :]))
            if checksum != 1:
                normalizer = 1 / checksum
                templates[i, :, :] = templates[i, :, :] * normalizer
            else:
                pass

        # insert src values into new array shape.
        for i in range(0,orig_shape[0]):
            for j in range(0,orig_shape[1]):
                newsrc[i,j] = src[i,j]
        #now, go through every character for original array and multiply the template by array at each position, store
        #the sums of these multiplications as a "goodness" metric, compare to threshold value.
        character_metrics = numpy.zeros((templates.shape[0],orig_shape[0],orig_shape[1]))
        boolean_array = numpy.zeros(character_metrics.shape) # this will convert the character metrics to their respective
        # histograms. iterate through all of the arrays and detect where the threshold is greater than the input
        # threshold for matching. every location where a metric lies above the threshold, the program inserts a 1 for
        # counting purposes.

        for a in range(0,templates.shape[0]):
            #this loop initiates for each template
            for i in range(0,character_metrics.shape[1]):
                for j in range(0, character_metrics.shape[2]):
                    #these loops go through the various positions in the image.
                    character_metrics[a,i,j] = sum(sum(newsrc[i:i+templates.shape[1],j:j+templates.shape[2]]*templates[a,:,:]))
                    if character_metrics[a,i,j] > threshold:
                        boolean_array[a,i,j] = 1
                    else:
                        boolean_array[a,i,j] = 0
    # Spatial Filtering Technique
    elif filterType == 'spatial':

        #The reason this method works on the individual characters is because when we unroll the image into a 1D array,
        #it's a perfect alignment with the character's template. When we unroll the source image in the same fashion,
        #due to the dimensions being different in the alphabet or text image, the 1D vectors no longer align.

        #we need to convert our image and filter shapes to 1D vectors.

        #append whitespace to end of the src array, such that we can iterate over original array length.
        if src.dtype and templates.dtype == numpy.uint8:
            src = src/255
            templates = templates/255

            #manually reshaping templates per my specification

            reshaped_templates = numpy.zeros((40,(templates.shape[1]*templates.shape[2])))
            count = 0
            for h in range(0,templates.shape[0]):
                for i in range(0,templates.shape[1]):
                    for j in range(0,templates.shape[2]):
                        reshaped_templates[h,count] =templates[h,i,j]
                        count = count+1
                count = 0



            # reshaped_templates_test = numpy.reshape(templates, [40, (templates.shape[1]*templates.shape[2])])
            # reshaped_src_test = numpy.ndarray.flatten(src)


            whitespace_horiz = numpy.zeros((src.shape[0],templates.shape[2]))+1
            reshaped_src = numpy.hstack((src, whitespace_horiz))
            whitespace_vert = numpy.zeros((templates.shape[1],reshaped_src.shape[1]))
            reshaped_src = numpy.vstack((reshaped_src,whitespace_vert))
        else:
            print("One or more of these images are not of dtype: 8-bit integer")
            exit()

        #create array to store all of the cosine values
        character_metrics = numpy.zeros((codes.size,orig_shape[0],orig_shape[1]))

        for h in range(0,codes.size):
            a = reshaped_templates[h,:]
            mag_a = numpy.sqrt((a*a).sum())
            for i in range(0,orig_shape[0]):
                for j in range(0,orig_shape[1]):
                    b = numpy.reshape(reshaped_src[i:i+templates.shape[1],j:j+templates.shape[2]],-1)
                    mag_b = numpy.sqrt((b*b).sum())
                    character_metrics[h,i,j] = numpy.dot(a, b)/(mag_a*mag_b)

        test = numpy.zeros(codes.size)
        for i in range(0, codes.size):
            test[i] = numpy.max(character_metrics[i, :])

        boolean_array = numpy.zeros(character_metrics.shape)

        # iterate through all of the arrays and detect where the threshold is greater than the input threshold for matching.
        # every location where a metric lies above the threshold, the program inserts a 1 for counting purposes

        for h in range(0, templates.shape[0]):
            # this loop initiates for each template
            for i in range(0, character_metrics.shape[1]):
                for j in range(0, character_metrics.shape[2]):
                    # these loops go through the various positions in the image.
                    if character_metrics[h, i, j] > threshold:
                        boolean_array[h, i, j] = 1
                    else:
                        boolean_array[h, i, j] = 0

        # count instances of each character
    array_sums = numpy.zeros(codes.size)
    for i in range(0, array_sums.size):
        array_sums[i] = sum(sum(boolean_array[i, :, :]))

    #create histogram of outputs
    fig,ax = plt.subplots(1,1)

    chrcodes = list(map(chr, codes))
    ax.plot(array_sums,'.')
    ax.set_title("Histogram of Character Counts")
    matplotlib.pyplot.xticks(numpy.linspace(0,codes.size,codes.size), chrcodes ,rotation = 0)  # Set text labels and properties.
    ax.set_ylabel('Iterations')
    ax.set_xlabel('Character')

    #print output text

    #create an array that converts the "hits" to their unicode character equivalents

    textalign = numpy.zeros(boolean_array.shape)
    for i in range(0,codes.size):
        textalign[i,:,:] = numpy.where((boolean_array[i,:,:] == 1),codes[i],0)

    #now sum the 40 unique character arrays into a single array, which will represent the displayed text.
    textalign = numpy.sum(textalign, (0))

    #reshape into a 1D array
    textalign = numpy.reshape(textalign, -1)

    #eliminate zeros
    textalign = textalign[textalign!=0].astype(int)

    #convert to letters, input into string
    textalign = list(map(chr, textalign))
    textalign =''.join(textalign)

    return textalign, fig