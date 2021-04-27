import math

import numpy as np
from scipy.interpolate import UnivariateSpline
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


def emp_performance_indicators(lmd, mu, reject_count, minutes_for_model, it_num):
    # Р_отк - вероятность отказа
    P = round(reject_count/(minutes_for_model * it_num), 3)
    # Q - относительная пропускная способность
    Q = round(1 - reject_count / (minutes_for_model * it_num), 3)
    # А - абсолютная пропускная способность СМО
    A = round(lmd * (1-reject_count/(minutes_for_model * it_num)), 3)
    # K_зан - среднее число занятых каналов
    K = round(lmd * (1-reject_count/(minutes_for_model * it_num)) / mu, 3)
    return [P, Q, A, K], \
           ['Failure probability', 'Relative throughput', 'Absolute throughput', 'Average number of busy channels']


def emp_lim_prob(p_pred):
    # p_pred[0] = p_pred[0] - 0.2
    # print(p_pred)
    # p_pred = p_pred / np.max(p_pred)
    print(p_pred)

    return [p_pred[i] / np.sum(p_pred) for i in range(len(p_pred))]


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


def smooth_solutions_by_spline(solutions, time):
    new_timespan = np.linspace(min(time), max(time), 1000)
    for i, sol in enumerate(solutions):
        solutions[i] = smooth_line_by_spline(time, sol, new_timespan)
    return new_timespan, solutions


def smooth_solutions(solutions, time):
    # new_timespan = np.linspace(min(time), max(time), 1000)
    for i, sol in enumerate(solutions):
        solutions[i] = smooth_line(time, sol, time)
    return time, solutions


def smooth_line_by_spline(list_x, list_y, new_timespan):
    spl = UnivariateSpline(list_x, list_y, k=3)
    return spl(new_timespan)


def smooth_line(list_x, list_y, new_timespan):
    # spl = UnivariateSpline(list_x, list_y, k=3)
    x_pred = 0
    for i, x in enumerate(list_y):
        if i == 0:
            x_pred = x
        else:
            list_y[i] = (x_pred + x) / 2
            x_pred = x
    return list_y

