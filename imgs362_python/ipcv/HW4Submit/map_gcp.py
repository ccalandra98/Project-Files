def map_gcp(src, map, srcX, srcY, mapX, mapY, order=1):
    # We need to form a system that defines a matrix for a homogenous system. We can then take the same approach as RST
    # by multipling the homogenous xy coordinates to get mapped coordinates.

    # model_matrix = x flat, independent_matrix = y flat.

    # The model matrix size is determined by order polynomial. It will be n columns as determined by the order, by m rows
    # which are predefined by the input data. I need to create a loop that fills a matrix of the proper size with polynomial
    # terms that can be used as predictors.
    import numpy
    import math
    from numpy import zeros
    from numpy import matmul
    from numpy import linspace
    from numpy import transpose
    from numpy import reshape
    from numpy import float32

    num_terms = int((order + 1) * ((order + 1) + 1) / 2)
    if isinstance(order, int) == False: #Ensure that order is passed as an integer
        print("Error: Order must be passed in as integer")
        exit()
    else:
        pass
    # check order
    if order <= 0:
        print("Error: Cannot have a 0th order or negative polynomial!")
        exit()
    else:
        model_matrix = zeros((len(srcX), num_terms + 1))
        # Create a loop that generates a list of exponents to be used as coefficients.
        max_x_exponent = order
        y_exponent = 0
        count = 0
        exponents = zeros((num_terms, 2)) #This code creates a list of exponents whos sums never exceed the order.
        for y in range(0, order + 1):
            for x in range(0, max_x_exponent + 1):
                exponents[count, 0] = int(x)
                exponents[count, 1] = int(y)
                count = count + 1
            max_x_exponent = max_x_exponent - 1
            y_exponent = y_exponent + 1
        exponents = exponents
        # Now apply these exponents to the source coordinates to create a model.
        for i in range(0, num_terms):
            for j in range(0, len(mapX)):
                model_matrix[j, i] = math.pow(mapX[j], exponents[i, 0]) * math.pow(mapY[j], exponents[i, 1])
                j = j + 1
            i = i + 1
        # appending matrix shape to get rid of unwanted last column of zeros
        matrix_shape = model_matrix.shape
        model_matrix = model_matrix[:, 0:matrix_shape[1] - 1]

    # 45:18 on 9/8/20's lecture

    # c_flat matrices are the coefficients that can be used to guess the locations of coordinates in the map.
    c_flat_y = matmul(
        matmul(numpy.linalg.inv(numpy.matmul(transpose(model_matrix), model_matrix)), transpose(model_matrix)), srcY)
    c_flat_x = matmul(
        matmul(numpy.linalg.inv(numpy.matmul(transpose(model_matrix), model_matrix)), transpose(model_matrix)), srcX)
    c_flat_array = numpy.vstack((c_flat_x, c_flat_y))

    # NO ERRORS FROM HERE UP
    # This part was taken from mapRST, creates the array shape and then the list of all possible indices.
    imshape = map.shape
    rows = linspace(0, imshape[0] - 1, imshape[0]).astype(int)
    columns = transpose(linspace(0, imshape[1] - 1, imshape[1]).astype(int))

    indices_array = numpy.zeros([imshape[0], imshape[1]])
    indices_array = transpose(numpy.reshape(indices_array, [1, int(indices_array.size)]).astype(int))
    indices_array = numpy.hstack((indices_array, indices_array))
    count = 0
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            indices_array[count, 1] = columns[j]
            indices_array[count, 0] = rows[i]
            count = count + 1

    # Now take these indices and apply them to the coefficients(index)^exponent to sum and get coordinates.

    x_matrix = zeros((indices_array.shape[0], num_terms))
    for i in range(0, indices_array.shape[0]):
        for j in range(0, num_terms):
            x_matrix[i, j] = math.pow(indices_array[i, 0], exponents[j, 0]) * math.pow(indices_array[i, 1],
                                                                                       exponents[j, 1])

    map1 = numpy.zeros(indices_array.shape[0])
    map2 = numpy.zeros(indices_array.shape[0])

    for i in range(0, indices_array.shape[0]):
        map1[i] = numpy.matmul(c_flat_array[1, :], transpose(x_matrix[i, :]))
        map2[i] = numpy.matmul(c_flat_array[0, :], transpose(x_matrix[i, :]))

    #these serve as checks for the output coefficents.
    y_flat_guess = numpy.dot(c_flat_array[1, :], transpose(model_matrix))
    x_flat_guess = numpy.dot(c_flat_array[0, :], transpose(model_matrix))

    map1 = reshape(map1, (map.shape[0], map.shape[1])).astype(float32)
    map2 = reshape(map2, (map.shape[0], map.shape[1])).astype(float32)
    return map1, map2
