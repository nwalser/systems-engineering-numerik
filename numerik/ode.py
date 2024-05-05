import numpy as np


def explicit_euler(x_end, h, x0, y0, f):
    xs = np.arange(x0, x_end + 1/2 * h, h, dtype=float)
    ys = np.zeros_like(xs, dtype=float)
    ys[0] = y0

    for i, x in enumerate(xs[:-1]):
        y = ys[i]
        r1 = f(x, y)
        r = r1
        ys[i+1] = y + r * h

    return np.array(xs), np.array(ys)


def explicit_runge(x_end, h, x0, y0, f):
    xs = np.arange(x0, x_end + 1/2 * h, h, dtype=float)
    ys = np.zeros_like(xs, dtype=float)
    ys[0] = y0

    for i, x in enumerate(xs[:-1]):
        y = ys[i]
        r1 = f(x, y)
        r2 = f(x+1/2*h, y+1/2*r1*h)
        r = 0*r1 + 1 * r2
        ys[i+1] = y + r * h

    return np.array(xs), np.array(ys)


def explicit_runge_kutta(x_end, h, x0, y0, f):
    xs = np.arange(x0, x_end + 1/2 * h, h, dtype=float)
    ys = np.zeros_like(xs, dtype=float)
    ys[0] = y0

    for i, x in enumerate(xs[:-1]):
        y = ys[i]
        r1 = f(x, y)
        r2 = f(x+1/2*h, y+1/2*r1*h)
        r3 = f(x+1/2*h, y+1/2*r2*h)
        r4 = f(x+1*h, y+1*r3*h)
        r = 1/6*r1 + 2/6 * r2 + 2/6 * r3 + 1/6 * r4
        ys[i+1] = y + r * h

    return np.array(xs), np.array(ys)


def explicit_heun(x_end, h, x0, y0, f):
    xs = np.arange(x0, x_end + 1/2 * h, h, dtype=float)
    ys = np.zeros_like(xs, dtype=float)
    ys[0] = y0

    for i, x in enumerate(xs[:-1]):
        y = ys[i]
        r1 = f(x, y)
        r2 = f(x+1*h, y+1*r1*h)
        r = 1/2*r1 + 1/2*r2
        ys[i+1] = y + r * h

    return np.array(xs), np.array(ys)


def implicit_euler(x_end, h, x0, y0, f, df):
    x = [x0]
    y = [y0]

    def G(s, xk, yk):
        return s - yk - h * f(xk, s)

    def dG(s, xk, yk):
        return 1 - h * df(xk, s)

    def newton(s, xk, yk, tol=1e-12, max_iter=20):
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


def implicit_midpoint(x_end, h, x0, y0, f, df):
    x = [x0]
    y = [y0]

    def G(yk1, xk, yk):
        return yk1 - (yk + h * f(xk + h/2, (yk+yk1)/2))

    def dG(yk1, xk, yk):
        return 1 - h * df(xk + h/2, (yk+yk1)/2)

    def newton(r, xk, yk, tol=1e-12, max_iter=20):
        for k in range(max_iter):
            delta = G(r, xk, yk)/dG(r, xk, yk)
            r -= delta
            if np.abs(delta) < tol:
                break
        return r

    while x[-1] < x_end-h/2:
        xk1 = x[-1]+h
        yk1 = newton(0, x[-1], y[-1])
        x.append(xk1)
        y.append(yk1)

    return np.array(x), np.array(y)


def implicit_trapezoid(x_end, h, x0, y0, f, df):
    x = [x0]
    y = [y0]

    def G(yk1, xk, yk):
        return yk1 - (yk + h/2 * (f(xk, yk) + f(xk+h, yk1)))

    def dG(yk1, xk, yk):
        return 1 - h/2 * (df(xk, yk) + df(xk+h, yk1))

    def newton(r, xk, yk, tol=1e-12, max_iter=20):
        for k in range(max_iter):
            delta = G(r, xk, yk)/dG(r, xk, yk)
            r -= delta
            if np.abs(delta) < tol:
                break
        return r

    while x[-1] < x_end-h/2:
        xk1 = x[-1]+h
        yk1 = newton(0, x[-1], y[-1])
        x.append(xk1)
        y.append(yk1)

    return np.array(x), np.array(y)
