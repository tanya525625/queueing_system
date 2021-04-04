import numpy as np
# import matplotlib.pyplot as plt

from queuing_system import QueuingSystem


def analyze_for_different_time(system: QueuingSystem, timespan: np.array, p_0: np.array, t_0: int):
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
        analytically_solutions.append(system.solve_analytically(p_0, t_0, t))
    analytically_solutions.insert(0, p_0)
    timespan.insert(0, t_0)
    analytically_solutions = list(np.transpose(analytically_solutions))
    # analytically_solutions = list(analytically_solutions)
    return analytically_solutions, timespan


def initial_conditions(n):
    p_0 = np.zeros(n + 1)
    p_0[0] = 1
    t_0 = 0
    return p_0, t_0


if __name__ == '__main__':
    n = 3
    lmd = 0.25
    mu = 1 / 3
    p_0, t_0 = initial_conditions(n)
    system = QueuingSystem(lmd, mu, n)
    timespan = list(range(1, 11))
    analytically_solutions, timespan = analyze_for_different_time(system, timespan, p_0, t_0)
    # plt.plot(analytically_solutions)
    # plt.show()
    system.make_plot(timespan, analytically_solutions)