#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: probs.py
#
#
#          DESCRIPTION: 
#
#
#                       RULE: 
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 07-04-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from .base import RewardSchedule


class SigmoidSchedule(RewardSchedule):
    """
    Sigmoid-based reward schedule for the rMIB experiment
    """
    def __init__(self, root, params):
        """
        Initialize the probability-based reward schedule
        """
        # Set attributes
        super().__init__(root, params)

    @staticmethod
    def sigmoid(choice_freq, inf_point, c, s, m):
        """
        Computes the value of a scaled sigmoid

        Args:
            choice_freq (float): The frequency value
            r (float): The lower bound of the sigmoid
            c (float): The center of the sigmoid
            s (float): The scale factor of the sigmoid
            m (float): The shift factor of the sigmoid

        Returns:
            float: The computed value of the scaled sigmoid
        """
        return 1 / (1 + np.exp(-(c - (choice_freq - inf_point)) / s)) - m


class GeneralSchedule(RewardSchedule):
    """
    Richard's curve reward schedule for the rMIB experiment
    """
    def __init__(self, root, params):
        """
        Initialize the probability-based reward schedule
        """
        # Set attributes
        super().__init__(root, params)

    def get_reward_probs(self, choice_prob):
        """
        Computes reward probabilities from the general curve based on frequency of choice for
        left (location informative) or down (feature informative).

        Args:
            choice_prob (float): Frequency of choosing left/down

        Returns:
            dict: reward probabilities (prob_left, prob_right, prob_down, prob_up)
        """
        rew_left = self.general_curve(
            choice_prob, 
            self.params["A"],
            self.params["K"],
            self.params["B"],
            self.params["v"],
            self.params["Q"],
            self.params["C"],
            self.params["M"]
        )
        rew_right = self.general_curve(
            choice_prob, 
            self.params["A"],
            self.params["K"],
            -1 * self.params["B"],
            self.params["v"],
            self.params["Q"],
            self.params["C"],
            self.params["M"]
        )
        return rew_left, rew_right

    def add_choice(self, choice):
        """
        Add a choice to the schedule
        """
        self.choices.append(choice)

    def get_incomes(self, choice_freq):
        """
        Compute the income of choosing the informative feature based on the Richards' curve.
        """
        pass

    def simulate(self, n_trials):
        """
        Simulate the reward schedule based on the Richards' curve.
        """
        pass

    def plot_schedule(self):
        """
        Plot the reward schedule
        """
        fig, (pax, rax) = plt.subplots(1, 2, figsize=(20, 6))
        schedule = self.get_schedule()

        # Plot probability
        pax.plot(schedule["choice_A"], schedule["prob_A"], label="P(A)")
        pax.plot(schedule["choice_B"], schedule["prob_B"], label="P(B)")
        pax.set_ylim(0, 1)
        pax.set_ylabel("Probability of reward")
        pax.set_xlabel("Frequency of choosing A (%)")
        pax.legend()

        # Plot reward rate
        rax.plot(schedule["choice_A"], schedule["rate_A"], label="Rate(A)")
        rax.plot(schedule["choice_B"], schedule["rate_B"], label="Rate(B)")
        rax.plot(schedule["choice_A"], schedule["rate_total"], label="Rate(Total)")
        rax.set_ylabel("Reward rate")
        rax.set_xlabel("Frequency of choosing A (%)")
        rax.legend()

        # Save
        save_file = self.fig_dir / f"{self.name}_{self.inf_dim}_{self.name}_v{self.version}"
        fig.savefig(save_file)

    @staticmethod
    def dick_curve(prob, A, K, B, v, Q, C, M):
        """
        Computes the value of the Richards' curve (generalized logistic function).

        Args:
            t (array-like): The input values where the curve is evaluated.
            A (float): Left horizontal asymptote.
            K (float): Right horizontal asymptote.
            B (float): The growth rate.
            v (float): Asymmetry of the curve.
            C (float): Affects the position of the curve along the y-axis.
            Q (float): Affects the position of the curve along the x-axis.
            M (float): Inflection point

        Returns:
            array-like: The computed values of the Richards' curve.
        """
        return A + (K - A) / (C + Q * np.exp(-B * (prob - M)))**(1 / v)

    def mse(self, free_params):
        """
        Computes the mean squared error of the Richard's curve at y=infliction_point.

        Args:
            free_params (tuple): Values from the optimization function

        Returns:
            float: The mean squared error of the Richard's curve
        """
        set_params = self.sch_params["GeneralizedLogistic"]["set_params"]
        inf_point = self.sch_params["infliction_point"]
        free_param_names = self.sch_params["GeneralizedLogistic"]["free_params"]
        free_params = {k: v for k, v in zip(free_param_names, free_params)}
        y = self.dick_curve(prob=inf_point, M=inf_point, **free_params, **set_params)
        return (inf_point - y) ** 2

    # def optimize(self, params):
    #     """
    #     Optimize the general curve by finding the free parameter values that minimize the MSE between
    #     the inflection point and the function at the same value.
    #     This is for the sake of matching behavior.

    #     Args:
    #         params (dict): A dictionary containing the parameters required to calculate the 'general' curve.

    #     Returns:
    #         float: The optimized parameter value.

    #     Raises:
    #         RuntimeError: If the optimization fails.
    #     """
    #     n_free = len(self.sch_params["GeneralizedLogistic"]["free_params"])
    #     init_guess = np.random.uniform(0, 1, size=n_free)
    #     results = minimize(self.mse, init_guess, method="Nelder-Mead")
        # if results.success:
        #     optp = {k: v for }
        # else:
        #     raise RuntimeError("Optimization failed.")
