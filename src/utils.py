import numpy as np


def matrix(lmd, mu, n):
    """
    Function for Kolmagorov matrix creation

    :param lmd: applications flow
    :param mu: service intensity
    :param n: channels count
    :return: matrix with differential equation coefficients
    """

    f = np.zeros([n + 1, n + 1])
    for i in range(n + 1):
        for j in range(n + 1):
            if j == i - 1:
                f[i, j] = lmd
            if j == i:
                f[i, j] = -(lmd + i * mu)
            if j == i + 1:
                f[i, j] = (i + 1) * mu
    return f
