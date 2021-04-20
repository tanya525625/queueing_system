import numpy as np

import utils
from queuing_system import AnalyticSystem, ImitationModel


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
    analytic_system.make_plot(timespan, analytically_solutions, 'analytic_solution', utils.performance_indicators,
                              (lmd, mu, analytically_solutions[-1]), utils.limit_prob, (n, lmd, mu))

    minutes_for_model = 100  # Количество генерируемых заявок)
    min_it = 10000
    it_num = 500
    max_t = 10
    h = 1  # Срез времени
    im_model = ImitationModel(lmd, mu, n)
    p, p_pred, min_it, reject_count = im_model.solve_by_imitation(minutes_for_model, min_it, it_num, h)
    time, solutions = im_model.filter_by_max_time(p, max_t, min_it, h)
    time, solutions = utils.smooth_solutions(solutions, time)
    im_model.make_plot(time, solutions, 'imitation_model',
                       utils.emp_performance_indicators, (lmd, mu, reject_count, minutes_for_model, it_num),
                       utils.emp_lim_prob, (p_pred, ))

