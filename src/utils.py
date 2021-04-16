import math

import numpy as np
import plotly.graph_objects as go


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
def performance_indicators(lmd, mu, p_n):
    # Р_отк - вероятность отказа
    P = round(p_n[-1], 3)
    # Q - относительная пропускная способность
    Q = round(1-P, 3)
    # А - абсолютная пропускная способность СМО
    A = round(lmd * Q, 3)
    # K_зан - среднее число занятых каналов
    K = round(A / mu, 2)
    return [P, Q, A, K], \
           ['Failure probability', 'Relative throughput', 'Absolute throughput', 'Average number of busy channels']


def make_layout(text):
    return go.Layout(
        annotations=[
            go.layout.Annotation(
                text=text,
                align='left',
                valign='middle',
                showarrow=True,
                xref='paper',
                yref='paper',
                x=1.14,
                y=0.01,
                bordercolor='black',
                borderwidth=1
            )
        ]
    )


def constants_as_legend_text(constants, names):
    return '<br>'.join([f"{name}: {const}" for name, const in zip(names, constants)])
