import numpy as np
from numerik import lu
from numerik import qr


def newton(f, df, x, tolerance=1e-14, max_step=100):
    step = 0
    error = np.linalg.norm(f(x))
    while error > tolerance and step < max_step:
        # print(step, x, error)
        x += lu.linsolve(df(x), -f(x))
        error = np.linalg.norm(f(x))
        step += 1

    return x


def gauss_newton(f, df, x, tolerance=1e-14, max_step=100):
    step = 0
    error = np.linalg.norm(f(x))
    while error > tolerance and step < max_step:
        #print(step, x, error)
        x += qr.linsolve_least_squares(df(x), -f(x))
        error = np.linalg.norm(f(x))
        step += 1

    return x
