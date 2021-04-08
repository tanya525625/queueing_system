import numpy as np


def F(f, p):
    F = np.dot(f, p)
    return F


def one_step(f, p, h):
    k1 = F(f, p)
    k2 = F(f, p + h / 2 * k1)
    k3 = F(f, p + h / 2 * k2)
    k4 = F(f, p + h * k3)
    return p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def optimization(prev_v, curr_v, eps=0.0001):
    return np.all(abs(prev_v - curr_v) < eps)


def rk4(f, t_0, p_0, t, h=0.1):
    tV = np.arange(t_0, t + h, h)
    tV[-1] = t
    p = np.empty((len(tV), len(p_0)))
    p[0] = p_0
    for i, index in enumerate(tV[:-2]):
        p[i + 1] = one_step(f, p[i], h)
        if optimization(p[i + 1], p[i]):
            p = p.tolist()[: i + 1]
            break
    h = tV[-1] - tV[-2]
    p[-1] = one_step(f, p[-2], h)
    return p[-1]
