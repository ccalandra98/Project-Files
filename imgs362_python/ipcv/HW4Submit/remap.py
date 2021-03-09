from ipcv.constants import INTER_NEAREST,INTER_CUBIC,INTER_LINEAR
from ipcv.constants import BORDER_WRAP,BORDER_REPLICATE,BORDER_REFLECT,BORDER_CONSTANT,BORDER_REFLECT_101
import numpy

def remap(src, map1, map2, interpolation=INTER_NEAREST, borderMode=BORDER_CONSTANT, borderValue=0):

    dst = numpy.zeros((map1.shape[0], map1.shape[1], 3))
    if interpolation == INTER_NEAREST:
        # start by rounding integers
        map1 = numpy.round(map1).astype(int)
        map2 = numpy.round(map2).astype(int)
        # This block of code finds all integers that fall outside of the boundary range and converts them to -1, so that
        # they can be targeted for border treatment.
        for r in range(0, dst.shape[0]):
            for c in range(0, dst.shape[1]):
                if map1[r, c] < 0:
                    map1[r, c] = -1
                if map1[r, c] >= src.shape[1]:
                    map1[r, c] = -1
                if map2[r, c] < 0:
                    map2[r, c] = -1
                if map2[r, c] >= src.shape[0]:
                    map2[r, c] = -1
    else:
        print("No other interpolation types defined yet.")
        exit()
    for r in range(0, dst.shape[0]):
        for c in range(0, dst.shape[1]):
            if (map1[r, c] >= 0) and (map2[r, c] >= 0):
                dst[r, c, :] = src[map2[r, c], map1[r, c], :]
            else:
                dst[r, c, :] = -1
    # We now have X and Y maps with positive indices and -1. These -1 are out of bounds, we now target them and give
    # them border treatments.
    if borderMode == BORDER_CONSTANT:
        #Find values equal to -1 and set them to the border value
        for i in range(0, dst.shape[0]):
            for j in range(0, dst.shape[1]):
                if any(dst[i, j, :] == -1):
                    dst[i, j, :] = borderValue
                else:
                    dst[i, j, :] = dst[i, j, :]
    elif borderMode == BORDER_REPLICATE:
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
    dst = dst.astype(numpy.uint8)
    return dst
