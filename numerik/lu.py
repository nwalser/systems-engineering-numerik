import numpy as np


def forward_substitution(L, b):
    # Get number of rows
    n = L.shape[0]

    # Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double)

    # Here we perform the forward-substitution.
    # Initializing  with the first row.
    y[0] = b[0] / L[0, 0]

    # Looping over rows in reverse (from the bottom  up),
    # starting with the second to last row, because  the
    # last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def back_substitution(U, y):
    # Number of rows
    n = U.shape[1]

    # Allocating space for the solution vector
    x = np.zeros_like(y, dtype=np.double)

    # Here we perform the back-substitution.
    # Initializing with the last row.
    x[-1] = y[-1] / U[-1, -1]

    # Looping over rows in reverse (from the bottom up),
    # starting with the second to last row, because the
    # last row solve was completed in the last step.
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    return x

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


# solves for Ax = b
def linsolve(a, b):
    lu = lu_decomposition(a)
    x = fb_subs(lu, b)
    return x
