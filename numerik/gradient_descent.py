import numpy as np
from numerik import lu


# newton
def newton(f, df, x, tolerance=1e-14, max_step=100):
    step = 0
    error = np.linalg.norm(f(x))
    while error > tolerance and step < max_step:
        # print(step, x, error)
        x += lu.linsolve(df(x), -f(x))
        error = np.linalg.norm(f(x))
        step += 1

    return x
