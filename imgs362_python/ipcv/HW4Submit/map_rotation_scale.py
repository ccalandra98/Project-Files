def map_rotation_scale(src, rotation=0, scale=[1, 1]):
    import numpy
    import math
    from math import cos, sin
    from numpy import linspace, transpose, ones
    # we need to create the 3x3 linear system to operate upon this homogenous array from start to finish, which will
    # contain both the scaling factor and the rotating factor.

    # create scaling matrix
    scaling_matrix = [(1 / scale[1]), 0, 0, 0, (1 / scale[0]), 0, 0, 0, 1]
    scaling_matrix = numpy.asarray(scaling_matrix)
    scaling_matrix = scaling_matrix.reshape((3, 3))

    # create rotation matrix
    rotation_matrix = [cos(math.radians(rotation)), sin(math.radians(rotation)), 0, -sin(math.radians(rotation)),
                       cos(math.radians(rotation)), 0, 0, 0, 1]
    rotation_matrix = numpy.asarray(rotation_matrix)
    rotation_matrix = rotation_matrix.reshape((3, 3))

    coordinate_matrix = numpy.matmul(rotation_matrix, scaling_matrix)

    # MATRIX IS CREATED, BUILD THE NEW ARRAY HERE
    imshape_src = src.shape
    coordinates = ((0, 0, 1), (0, imshape_src[1], 1), (imshape_src[0], 0, 1), (imshape_src[0], imshape_src[1], 1))
    coordinates = transpose(numpy.array(coordinates))
    map_corners = numpy.dot(numpy.linalg.inv(coordinate_matrix), coordinates)

    map_length = round(max(map_corners[0, :]) - min(map_corners[0, :]))
    map_height = round(max(map_corners[1, :]) - min(map_corners[1, :]))
    imshape = (map_length, map_height)

    # Create list of indices
    rows = linspace(0, imshape[0] - 1, imshape[0]).astype(int)
    columns = transpose(linspace(0, imshape[1] - 1, imshape[1]).astype(int))

    # now, create an array of every possible index in source image.
    indices_array = numpy.zeros([map_length, map_height])
    indices_array = transpose(numpy.reshape(indices_array, [1, int(indices_array.size)]).astype(int))
    indices_array = numpy.hstack((indices_array, indices_array))
    count = 0
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            indices_array[count, 1] = columns[j]
            indices_array[count, 0] = rows[i]
            count = count + 1

    # append array of ones to indices array to create homogenous array
    ones_array = ones(indices_array.shape[0])
    homogenous_array = numpy.vstack((indices_array[:, 0], indices_array[:, 1], ones_array))

    # we are ready to apply this transform. Let's shift the coordinate system such that 0,0 is about the center.

    homogenous_array[0, :] = homogenous_array[0, :] - (imshape[0] / 2)
    homogenous_array[1, :] = homogenous_array[1, :] - (imshape[1] / 2)

    # Take the dot product of these matrices
    prime_array = numpy.dot(coordinate_matrix, homogenous_array)

    # now to shift back to upper left corner of image being (0,0)
    prime_array[0, :] = prime_array[0, :] + (imshape_src[0] / 2)
    prime_array[1, :] = prime_array[1, :] + (imshape_src[1] / 2)

    xprime = prime_array[0, :]
    yprime = prime_array[1, :]

    # Now have to convert these 1xn arrays to their destination sizes desired.

    xprime = xprime.reshape(imshape[0], imshape[1]).astype(numpy.float32)
    yprime = yprime.reshape(imshape[0], imshape[1]).astype(numpy.float32)

    return yprime, xprime
