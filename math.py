def dot_product(x, w):
    """
    This function calculates the dot product of vectors x and w

    :param x: vector
    :param w: vector
    :return: float
    """

    if len(x) != len(w):
        raise ValueError("arguments must have equal length")

    result = 0.0
    for j in range(len(w)):
        result += x[j] * w[j]
    return result


def matrix_multiply(X, Y):
    """
    This function returns the result of the matrix multiplication between X and Y
    X is a 2 dimensional array, Y is a vector
    The second dimension of X must be the same size as the first dimension of Y
    :param X: 2Dimensional arraay
    :param Y: vector
    :return: vector
    """
    if len(X[0]) != len(Y):
        raise ValueError("second dimension of X does not equal first dimension of Y")

    result = [0 for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            result[i] += X[i][j] * Y[j]

    return result


def transpose(X):
    """
    This function returnss the transpose of the given 2-dimensional array
    :param X: 2Dimensional matrix
    :return: transpose of X
    """
    w, h = X.shape
    result = [[0 for i in range(w)] for j in range(h)]
    for i in range(h):
        for j in range(w):
            result[i][j] = X[j][i]

    return result




