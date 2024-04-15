import numpy as np


# linear solve functions
def fb_subs(lu, b):
    (rows, columns) = lu.shape

    # solve LZ = b
    z = np.full(rows, 0, dtype=np.number)
    for row in range(rows):
        sub = np.dot(lu[row], z)
        z[row] = b[row] - sub

    # solve RX = Z
    x = np.full(rows, 0, dtype=np.number)
    for row in reversed(range(rows)):
        sub = np.dot(lu[row], x)
        x[row] = (z[row] - sub) / lu[row, row]

    return x


def lu_decomposition(a):
    (rows, columns) = a.shape

    # decompose inplace into LU matrix
    for column in range(columns):
        for row in range(column + 1, columns):
            scalar = a[row, column] / a[column, column]
            a[row, column:columns] -= (a[column] * scalar)[column:columns]
            a[row, column] = scalar

    return a


def linsolve(A, b):
    lu = lu_decomposition(A)
    x = fb_subs(lu, b)
    return x
