def find_nearest(input_array, value,
                 returnIndex=False):  # I found this function on stackoverflow, linking here for transparency:
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    import numpy
    from numpy import isnan

    input_array = numpy.asarray(input_array)  # I modified this code slightly to replace nan with 0
    location_nans = isnan(input_array)
    input_array[location_nans] = 0
    idx = (numpy.abs(
        input_array - value)).argmin()  # The function takes an array, and searches for the closest index to the
    # value given.
    if not returnIndex:
        return input_array[idx]
    else:
        return idx
