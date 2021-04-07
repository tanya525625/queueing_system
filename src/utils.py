import numpy as np
import math


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
                if i == n:
                    f[i, j] = -i * mu
                else:
                    f[i, j] = -(lmd + i * mu)
            if j == i + 1:
                f[i, j] = (i + 1) * mu
    return f


# Предельные вероятности, отобразить на графике в виде точек в конечной точке временного интервала
def limit_prob(n, lmd, mu):
    limit = np.zeros(n + 1)
    for i in range(n + 1):
        limit[0] += (lmd ** i) / (math.factorial(i) * mu ** i)
    for i in range(1, n + 1):
        limit[i] = ((lmd ** i) / (math.factorial(i) * mu ** i)) / limit[0]
    limit[0] = (limit[0]) ** -1
    return limit


# Показатели эффективности, отбразить в легенде желательно с названиями
def performance_indicators(n, lmd, mu):
    #     А - абсолютная пропускная способность СМО
    A = (lmd * mu) / (lmd + mu)
    # Q - относительная пропускная способность
    Q = mu / (lmd + mu)
    # Р_отк - вероятность отказа
    P = lmd / (lmd + mu)
    # K_зан - среднее число занятых каналов
    K = A / mu
    return [A, Q, P, K]
