import random

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import utils
from numerecal_methods import rk4


class QueuingSystem:
    """
    Queuing system with refusals, which has 2 approaches for getting
    failure distributions probabilities:
    1) Analytic (using Kolmagorov equations)
    2) Simulation modeling
    """

    def __init__(self, lmd, mu, n):
        self.lmd = lmd
        self.mu = mu
        self.n = n

    def make_plot(self, t, states_arr):
        fig = go.Figure()
        fig.update_layout(
            title="Failure probability distribution",
            xaxis_title="timespan (t)",
            yaxis_title="probability distributions values (p)",
        )
        probabilities_limits = utils.limit_prob(self.n, self.lmd, self.mu)
        colors = px.colors.sequential.Sunset[-(self.n + 1) :]
        for i, (state, prob_l, color) in enumerate(
            zip(states_arr, probabilities_limits, colors)
        ):
            state_name = f"P_{i}"
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=state,
                    name=f"{state_name} failure probability distribution",
                    fillcolor=color,
                    text=f"Failure probability distribution " f"in the p_{i} state",
                    line=dict(color=color, width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[t[-1]],
                    y=[prob_l],
                    mode="markers",
                    name=f"{state_name} probability limit",
                    text=f"{state_name} probability limit",
                    marker=dict(
                        size=10, color=color, line=dict(width=2, color="whitesmoke")
                    ),
                )
            )
        performance_indicators, perfomances_names = utils.performance_indicators(self.lmd, self.mu, states_arr[-1])
        fig.update_layout(utils.make_layout(utils.constants_as_legend_text(performance_indicators, perfomances_names)))
        fig.write_html("1.html")


class AnalyticSystem(QueuingSystem):
    def __init__(self, lmd, mu, n):
        QueuingSystem.__init__(self, lmd, mu, n)
        self.p_0, self.t_0 = self._initial_conditions()

    def _initial_conditions(self):
        p_0 = np.zeros(self.n + 1)
        p_0[0] = 1
        t_0 = 0
        return p_0, t_0

    def solve_analytically(self, t, h=0.01):
        matrix = utils.matrix(self.lmd, self.mu, self.n)
        return rk4(matrix, self.t_0, self.p_0, t, h)


class ImitationModel(QueuingSystem):
    def __init__(self, lmd, mu, n):
        self.lmd = lmd
        self.mu = mu
        self.n = n

    def solve_by_imitation(self, minutes_for_model, min_it, it_num, h, max_t):
        p = np.zeros((minutes_for_model - self.n - 1, self.n + 1))
        p_pred = np.zeros(self.n + 1)
        reject_count = 0
        for i in range(it_num):
            requests = []  # время получения заявок
            last_requests_time = 0.0
            for minute in range(0, minutes_for_model - 1):
                rnd = random.expovariate(self.lmd)  # Время между заявками
                last_requests_time += rnd
                requests.append(last_requests_time)
            if requests[-1] < min_it:
                min_it = requests[-1]
            print(requests)

            handles = []
            # Для каждой заявки генерируем время ее обработки
            for request in requests:
                rnd = random.expovariate(self.mu)
                handles.append(rnd)
            # print(handles)

            handles_end = np.zeros(self.n)  # вектор времени выхода заявки

            j = 0
            t = 0
            for i in range(self.n):
                handles_end[i] = requests[i] + handles[i]
            for request in range(self.n, len(requests)):
                k = 0  # счетчик занятых каналов
                for i in range(self.n):
                    if handles_end[i] > requests[request]:
                        k += 1
                    p_pred[k] += 1
                c = 0  # счетчик занятых каналов
                if requests[request] > t:
                    for i in range(self.n):
                        if handles_end[i] > requests[request]:
                            c += 1
                    p[j, c] += 1
                    j += 1
                    t += h

                ind = np.argmin(handles_end)
                if requests[request] < min(handles_end):
                    reject_count += 1  # отказано в обслуживании
                else:
                    handles_end[ind] = requests[request] + handles[request]
        return p, p_pred, min_it