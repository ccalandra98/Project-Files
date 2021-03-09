from ipcv.constants import BORDER_WRAP,BORDER_REPLICATE,BORDER_REFLECT,BORDER_CONSTANT,BORDER_REFLECT_101

def filter2D(src, dstDepth, kernel, delta=0, maxCount=255, borderMode = None, borderValue = 0):
    import numpy
    src_shape = tuple(src.shape)

    #assume source image is a nxmx3 array, if not, convert as such
    if len(src_shape) < 3:
        #this is an nxmx1 array, convert to 3-channel.
        src = numpy.stack((src,src,src),2)

    #check symmetry of kernel
    if kernel.shape[0]!= kernel.shape[1]:
        print("ERROR: Kernel must be symmetrical.")
        exit()
    if (kernel.shape[0] % 2) == 0 or (kernel.shape[1] % 2) == 0:
        print("ERROR: Kernel must have odd shape.")
        exit()

    #Now check for kernel weight, set equal to one if not already
    if sum(sum(kernel))!=1:
        checksum = sum(sum(kernel))
        if sum(sum(kernel)) == 0:
            pass #can't divide by zero, can't create a normalizing coeficcient if the sum is zero.
        else:
            normalizer = 1/checksum
            kernel = kernel*normalizer

    #We need to append values on top, bottom, and on the sides of the array to account for blank space in the image.
    array_offset = int((kernel.shape[0]-1)/2) #creates the offset, the size of which pads our array
    npad = ((array_offset, array_offset),(array_offset,array_offset),(0,0)) #specifies the dimensions along which to pad.

    if borderMode == None:
        dst = numpy.pad(src,npad ,'constant', constant_values=(0,0)) #simply pad with zeros. Default argument, fastest.

    elif borderMode == BORDER_CONSTANT:
        dst = numpy.pad(src, npad, 'constant', constant_values=(-1, -1))
        #Find values equal to -1 and set them to the border value
        for i in range(0, dst.shape[0]):
            for j in range(0, dst.shape[1]):
                if any(dst[i, j, :] == -1):
                    dst[i, j, :] = borderValue
                else:
                    dst[i, j, :] = dst[i, j, :]

    elif borderMode == BORDER_REPLICATE:
        dst = numpy.pad(src, npad, 'constant', constant_values=(-1, -1))
        # This code takes the first line of DST, finds the elements not equal to -1, and then sets the first and last element
        # in that line and uses them as the replication point.
        for i in range(0, dst.shape[0]):
            line_array = dst[i,:,0]+1
            idx = numpy.array(numpy.where(line_array !=0))
            idx_size = idx.size
            if idx_size ==0:
                pass    #this check is to prevent an error for index arrays with no values above -1.
            else:
                line_min = numpy.amin(idx)
                line_max = numpy.amax(idx)
                if line_max>=dst.shape[1]:
                    line_max =dst.shape[1]-1
                # FIND NEAREST NON NEGATIVE VALUE STARTS HERE
                for j in range(0, dst.shape[1]):
                    if any(dst[i, j, :] == -1):
                        if j<= line_min:
                            dst[i, j, :] = dst[i, line_min, :]
                        else:
                            dst[i, j, :] = dst[i, line_max, :]
                    else:
                        dst[i, j, :] = dst[i, j, :]
    # scale array back down to maximum of 1
    src = src / maxCount

    #initiate dst as array of zeros
    dst = numpy.zeros(dst.shape)
    count = 0   # For each position in the kernel, we now multiply the source by the kernel value, and shift it based
    # on the kernel value's position in the kernel array.

    for columns in range(0,kernel.shape[0]):
        for rows in range(0,kernel.shape[1]):
            temp_array = src*kernel[columns,rows] #temporarily stores current array calculation
            dst[columns:src_shape[0]+columns,rows:src_shape[1]+rows,:] =\
                dst[columns:src_shape[0]+columns,rows:src_shape[1]+rows,:]+temp_array #appends prior array calculation
                        # to total dst array, which is a sum of every cardinal position in the kernel.
            count = count+1

    #re-convert dst to 0-255 scale, set type per input specification, clip out the padded section.
    dst = ((dst[array_offset:src_shape[0]+array_offset,array_offset:src_shape[1]+array_offset,:]*maxCount)+delta)\
        .astype(dstDepth)
    return dst
