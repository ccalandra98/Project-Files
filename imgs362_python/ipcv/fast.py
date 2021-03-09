import numpy
from numpy import array
from numpy import sum as sums
from numpy import roll
from numpy import zeros
from numpy import vstack

def fast(src, differenceThreshold=50, contiguousThreshold=12, nonMaximalSuppression=True):

    orig_shape = src.shape

    # pad the array with copies
    src = numpy.pad(src, (3,3), 'edge').astype(float)

    # initiate candidacy arrays
    candidacy = numpy.zeros(orig_shape)
    candidacy_secondpass = numpy.zeros(orig_shape)

    for i in range(3, orig_shape[0] + 3):
        for j in range(3, orig_shape[1] + 3):
            #store current point, create initial cardinal points window.
            current = src[i, j]
            window = src[i - 3:i + 4, j - 3:j + 4]
            cardinals = array((window[3, 0], window[6, 3], window[3, 6], window[0, 3]))

            # count the cardinal points that exceed the difference threshold
            count = 0
            for k in range(0, cardinals.size):
                if current + differenceThreshold < cardinals[k] or current - differenceThreshold > cardinals[k]:
                    count = count + 1
                else:
                    count = count
            # if that count ==3, set as candidate
            if count == 3:
                candidacy[i - 3, j - 3] = 1
            else:
                candidacy[i - 3, j - 3] = 0


    # initialize variables for second pass
    vector = zeros(16)
    posvec = zeros(vector.size)
    negvec = zeros(vector.size)
    for i in range(3, orig_shape[0] + 3):
        for j in range(3, orig_shape[1] + 3):

            #check for candidacy
            if candidacy[i-3,j-3] != 1:
                pass
            else:
                #set current window and circular vector
                current = src[i,j]
                window = src[i - 3:i + 4, j - 3:j + 4]
                vector = array([window[3, 0], window[4, 0], window[5, 1], window[6, 2], window[6, 3], window[6, 4],
                                    window[5, 5],
                                    window[4, 6], window[3, 6], window[2, 6], window[1, 5], window[0, 4], window[0, 3],
                                    window[0, 2],
                                    window[1, 1], window[2, 0]])
                # determine whether these differences are positive or negative differences, check to see if they exceed
                # the threshold. If so, they are stored as a 1. If not, they are stored as a 0.
                for k in range(0, len(vector)):
                    if vector[k]-current >0 and abs(vector[k]-current) > differenceThreshold:
                        posvec[k] = 1
                        negvec[k] = 0
                    elif vector[k]-current <0 and abs(vector[k]-current) > differenceThreshold:
                        posvec[k] = 0
                        negvec[k] = 1
                    else:
                        posvec[k] = 0
                        negvec[k] = 0
                # Check to see if the vector sums have the potential to exceed the threshold
                if sums(posvec) >= contiguousThreshold or sums(negvec) >= contiguousThreshold:

                    #create arrays to avoid looping. arrays contain every possible arrangement of the 16 values
                    posarray = array(vstack((posvec, roll(posvec, 1), roll(posvec, 2), roll(posvec, 3), roll(posvec, 4),
                                             roll(posvec, 5), roll(posvec, 6), roll(posvec, 7), roll(posvec, 8),
                                             roll(posvec, 9), roll(posvec, 10), roll(posvec, 11), roll(posvec, 12),
                                             roll(posvec, 13), roll(posvec, 14), roll(posvec, 15))))

                    negarray = array(vstack((negvec, roll(negvec, 1), roll(negvec, 2), roll(negvec, 3), roll(negvec, 4),
                                             roll(negvec, 5), roll(negvec, 6), roll(negvec, 7), roll(negvec, 8),
                                             roll(negvec, 9), roll(negvec, 10), roll(negvec, 11), roll(negvec, 12),
                                             roll(negvec, 13), roll(negvec, 14), roll(negvec, 15))))

                    #now cut off ends of arrays using the threshold, then sum.
                    possums = sums(posarray[:, 0:contiguousThreshold], 1).astype(int)
                    negsums = sums(negarray[:, 0:contiguousThreshold], 1).astype(int)

                    # pause = 0
                    # Check to see if any of the arrangements of sums equal the threshold (which would indicate
                    # contiguous arrangement of points).
                    if any(possums == contiguousThreshold) or any(negsums == contiguousThreshold):
                        # hits
                        candidacy_secondpass[i-3,j-3] = 1
                    else:
                        candidacy_secondpass[i-3,j-3] = 0
                        # misses
                else:
                    candidacy_secondpass[i - 3, j - 3] = 0
                count = count+1

        # optional implementation of non-maximal suppression


    return candidacy_secondpass