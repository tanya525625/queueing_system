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

    def solve_analytically(self, p_0, t_0, t, h=0.01):
        matrix = utils.matrix(self.lmd, self.mu, self.n)
        return rk4(matrix, t_0, p_0, t, h)

    @staticmethod
    def make_plot(t, states_arr):
        fig = go.Figure()
        fig.update_layout(
            title="Failure probability distribution",
            xaxis_title="timespan (t)",
            yaxis_title="probability distributions values (p)",
        )
        for i, state in enumerate(states_arr):
            state_name = f"p_{i}"
            fig.add_trace(go.Scatter(x=t, y=state, name=state_name, text=f"Failure probability distribution "
                                                                         f"in the p_{i} state"))
        fig.show()




