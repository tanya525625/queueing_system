import numpy as np

# import matplotlib.pyplot as plt

import utils
from queuing_system import AnalyticSystem


def analyze_for_different_time(analytic_system: AnalyticSystem, timespan: np.array):
    """
    Function for launching analysis of the failure
    distributions by 2 approaches:
    1) Analytic (using Kolmagorov equations)
    2) Simulation modeling

    :param system: QueuingSystem instance
    :param timespan: timespan for analysis
    :param p_0: initial values of the zero state
    :param t_0: initial value of the time
    :return: solutions and timespan
    """

    analytically_solutions = []
    for t in timespan:
        analytically_solutions.append(analytic_system.solve_analytically(t))
    analytically_solutions.insert(0, analytic_system.p_0)
    timespan.insert(0, analytic_system.t_0)
    analytically_solutions = list(np.transpose(analytically_solutions))
    return analytically_solutions, timespan


if __name__ == "__main__":
    n = 3
    lmd = 0.25
    mu = 1 / 3

    analytic_system = AnalyticSystem(lmd, mu, n)
    timespan = list(range(1, 11))
    analytically_solutions, timespan = analyze_for_different_time(
        analytic_system, timespan
    )
    # plt.plot(analytically_solutions)
    # plt.show()
    analytic_system.make_plot(timespan, analytically_solutions)
