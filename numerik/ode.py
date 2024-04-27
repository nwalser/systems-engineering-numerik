import numpy as np


def explicit_euler(x_end, h, x0, y0, f):
    xs = np.arange(x0, x_end, h, dtype = float)
    ys = np.zeros_like(xs, dtype=float)
    ys[0] = y0

    for i, x in enumerate(xs[:-1]):
        y = ys[i]
        slope = f(x, y)
        ys[i+1] = y + slope * h

    return np.array(xs), np.array(ys)


def implicit_euler(x_end, h, x0, y0, f, df):
    x = [x0]
    y = [y0]

    def G(s, xk, yk):
        return s - yk - h * f(xk, s)

    def dG(s, xk, yk):
        return 1 - h * df(xk, s)

    def newton(s, xk, yk, tol=1e-12, max_iter=20):
        delta = 10*tol
        for k in range(max_iter):
            delta = G(s, xk, yk)/dG(s, xk, yk)
            s -= delta
            if np.abs(delta) < tol:
                break
        return s

    while x[-1] < x_end-h/2:
        y.append(newton(y[-1], x[-1], y[-1]))
        x.append(x[-1]+h)
    return np.array(x), np.array(y)
