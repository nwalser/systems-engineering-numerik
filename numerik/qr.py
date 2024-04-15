import numpy as np


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def householder(w):
    v = np.zeros_like(w)

    v[0] = w[0] + sign(w[0]) * np.linalg.norm(w)
    v[1:] = w[1:]

    H = np.eye(len(w)) - 2 * np.outer(v, v) / np.dot(v, v)

    return H


def qr_decomposition(a):
    (rows, columns) = a.shape

    q = np.identity(rows)
    r = np.copy(a)

    for column in range(columns):
        # get needed vector and apply householder transformation
        y = r[column:, column]
        hk = householder(y)

        # fill up into correct matrix dimension
        qk = np.identity(rows)
        qk[column:,column:] = hk

        # apply to matrices
        q = qk@q
        r = qk@r

    # cut into needed shapes
    r = r[0:columns,:]
    q = q.T[:,0:columns]

    return q, r


# solves Ax = b for x, least squares method with qr decomposition
def linsolve_least_squares(a, b):
    q, r = qr_decomposition(a)

    # backward substitution
    # solve Rx = Q.T * b
    (rows, columns) = r.shape

    z = q.T@b
    x = np.full(rows, 0, dtype=np.number)
    for row in reversed(range(rows)):
        sub = np.dot(r[row], x)
        x[row] = (z[row] - sub)/r[row, row]

    return x
