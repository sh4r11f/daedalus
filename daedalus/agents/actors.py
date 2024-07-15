#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: doers.py
#
#
#               DESCRIPTION: Doers do things.
#
#
#                      RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 07-10-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from daedalus.utils import moving_average


class Tailor:
    """
    A class to train agents.

    Args:
        agent (object): an agent object
        env (object): an env object
        n_episodes (int): number of episodes
    """
    def __init__(self, agent=None):
        """
        Initialize the trainer with an agent and an environment.
        """
        self._agent = agent
        self.results = []
        self.simulations = []
        self.trained = False

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, agent):
        self._agent = agent

    def reset(self):
        """
        Reset the trainer.
        """
        self.results = []
        self.simulations = []
        self.trained = False
        self.agent.reset()

    def fit(self, data, n_epochs=10, method="Nelder-Mead", bounded=False, verbose=False):
        """
        Train the agent for a specified number of episodes.
        """
        # np.random.seed(11)
        init_mult = 1
        for ep in range(n_epochs):

            # Initialize the parameters
            if not bounded:
                init_params = [np.random.uniform(0, 1) for _ in range(len(self._agent.params))]
            else:
                init_params = []
                for par, (lowerb, upperb) in self._agent.bounds:
                    if lowerb == -np.inf:
                        lowerb = -1e10
                    if upperb == np.inf:
                        upperb = 1e10
                    rand = np.random.uniform(lowerb, upperb) / init_mult
                    rand = np.clip(rand, lowerb, upperb)
                    init_params.append(rand)

            # Start the optimization
            self._agent.reset()
            results = minimize(
                self._agent.loss,
                init_params,
                args=(data,),
                method=method,
                # options={"disp": True if verbose else False, "maxiter": 1000},
                bounds=[bound[1] for bound in self._agent.bounds] if bounded else None,
            )
            self.results.append(results)
            if verbose:
                print("Epoch: ", ep + 1, " Loss: ", results.fun)
                print("- - " * 10)

        best_params = self.optimal_params()
        if best_params is not None:
            self._agent.params = best_params
            self.trained = True
        else:
            print("No best parameters found.")

        return best_params

    def show(self):
        """
        Display the results of the training.
        """
        for result in self.results:
            print(result)

    def average_loss(self):
        """
        Compute the average loss of the agent.
        """
        return np.nanmean([result.fun for result in self.results])

    def min_loss(self):
        """
        Compute the minimum loss of the agent.
        """
        return np.nanmin([result.fun for result in self.results])

    def max_loss(self):
        """
        Compute the maximum loss of the agent.
        """
        return np.nanmax([result.fun for result in self.results])

    def mcfadden_r2(self, log_likelihood, n):
        """
        Compute the McFadden R2 of the agent.
        """
        return 1 - (log_likelihood / (n * np.log(0.5)))

    def optimal_params(self):
        """
        Get the optimal parameters of the agent.
        """
        # find the best parameters
        best_loss = np.inf
        best_params = None
        for result in self.results:
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x

        return best_params

    def aic(self, log_likelihood):
        """
        Calculate the Akaike Information Criterion (AIC).

        Args:
            log_likelihood (float): The log-likelihood of the model.

        Returns:
            float: The AIC value.
        """
        return 2 * len(self.agent.params) - 2 * log_likelihood

    def bic(self, log_likelihood, num_observations):
        """
        Calculate the Bayesian Information Criterion (BIC).

        Args:
            log_likelihood (float): The log-likelihood of the model.
            num_observations (int): The number of observations in the dataset.

        Returns:
            float: The BIC value.
        """
        return len(self.agent.params) * np.log(num_observations) - 2 * log_likelihood

    def aic_p(self, log_likelihood, num_observations):
        """
        Calculate the corrected Akaike Information Criterion (AICc).

        Args:
            log_likelihood (float): The log-likelihood of the model.
            num_observations (int): The number of observations in the dataset.

        Returns:
            float: The AICc value.
        """
        return 2 * len(self.agent.params) / num_observations - 2 * log_likelihood

    def bic_p(self, log_likelihood, num_observations):
        """
        Calculate the corrected Bayesian Information Criterion (BICc).

        Args:
            log_likelihood (float): The log-likelihood of the model.
            num_observations (int): The number of observations in the dataset.

        Returns:
            float: The BICc value.
        """
        return 2 * len(self.agent.params) * np.log(num_observations) / num_observations - 2 * log_likelihood

    def exam(self, data, params, n_episodes=10, n_samples=None, sample_strategy="random"):
        """
        Simulate the agent for a specified number of episodes.
        """
        # np.random.seed(11)
        for _ in range(n_episodes):

            # Reset the agent and set the parameters
            self._agent.reset()
            self._agent.params = params

            # Sample the data
            if n_samples is not None:
                if sample_strategy == "consecutive":
                    sample = self.sample_consecutive(data, n_samples)
                else:
                    sample = self.sample(data, n_samples)
            else:
                sample = data

            # Simulate the agent
            for trial in sample:
                action = self._agent.choose_action()
                reward = trial[action]
                self._agent.update(action, reward)

            # Save the simulation
            self.simulations.append(
                {
                    "choices": self._agent.choices,
                    "rewards": self._agent.rewards,
                    "history": self._agent.history
                }
            )

    def sample_consecutive(self, array, sample_size):
        # Ensure sample_size is valid
        if sample_size > len(array):
            raise ValueError("Sample size cannot be larger than the array length")

        # Randomly choose a starting index such that the segment is within bounds
        start_index = np.random.randint(0, len(array) - sample_size + 1)

        # Slice the array to get the consecutive segment
        sampled_array = array[start_index:start_index + sample_size]
        return sampled_array

    def sample(self, data, n_samples):
        """
        Sample the data for the simulation.
        """
        data = np.array(data)
        indices = np.random.choice(data.shape[0], n_samples, replace=True)
        return data[indices]

    def add_sim(self, choices, rewards, history):
        """
        Add a simulation to the list of simulations.
        """
        self.simulations.append({
            "choices": choices,
            "rewards": rewards,
            "history": history
            })

    def plot_sim(self, num, title, window=30):
        """
        Plot the history of the agent.
        """
        sim = self.simulations[num]
        fig, ax = plt.subplots(2, 1, figsize=(24, 8))
        q_values = np.array(sim["history"])
        trials = np.arange(1, q_values.shape[0] + 1)
        ax[0].scatter(trials, q_values[:, 0], label=r'$Q_0$', c='#08a4a7', alpha=0.5)
        ax[0].scatter(trials, q_values[:, 1], label=r'$Q_1$', c='#FF00BF', alpha=0.5)
        ax[0].set_xlabel('Trial')
        ax[0].set_ylabel('Q-value')
        # ax.set_ylim(-1.1, 1.1)
        ax[0].set_title(title)
        ax[0].legend(loc='upper left')

        choices = np.array(sim["choices"])
        cma = moving_average(choices, window)
        rewards = np.array(sim["rewards"])
        rma = moving_average(rewards, window)
        ax[1].plot(trials, 1 - cma, label=r'Choice = 0', c='b', lw=3)
        ax[1].plot(trials, cma, label=r'Choice = 1', c='r', lw=3)
        ax[1].plot(trials, rma, label='Reward', c='#00B800', lw=3)
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel('Trial')
        ax[1].set_ylabel('Prob.')
        ax[1].set_title(title)
        ax[1].legend(loc="upper left")

        fig.tight_layout()

        return fig

    def save(self):
        """
        Save the simulations to a file.
        """
        pass


class Coach(Tailor):

    def __init__(self, agent=None):
        super().__init__(agent)

    def exam(self, data, params, n_episodes=10, n_samples=None, sample_strategy="random"):
        """
        Simulate the agent for a specified number of episodes.
        """
        # np.random.seed(11)
        for _ in range(n_episodes):

            # Sample the data
            if n_samples is not None:
                if sample_strategy == "consecutive":
                    sample = self.sample_consecutive(data, n_samples)
                else:
                    sample = self.sample(data, n_samples)
            else:
                sample = data

            # Reset the agent and set the parameters
            self._agent.reset()
            self._agent.params = params

            # Simulate the agent
            for cor_act, cor_feat, feat_left, rew in sample:
                action_choice, feature_choice = self._agent.choose_action(feat_left)
                if action_choice == cor_act and feature_choice == cor_feat:
                    reward = rew
                else:
                    reward = 0
                self._agent.update(action_choice, feature_choice, reward)

            # Save the simulation
            self.simulations.append(
                {
                    "action_choices": self._agent.action_choices,
                    "feature_choices": self._agent.feature_choices,
                    "rewards": self._agent.rewards,
                    "action_history": self._agent.action_history,
                    "feature_history": self._agent.feature_history,
                    "combined_history": self._agent.combined_history,
                }
            )

    def plot_multi_sim(self, num, title, window=30):
        """
        Plot the history of the agent.
        """
        sim = self.simulations[num]
        fig, ax = plt.subplots(2, 1, figsize=(24, 8))
        q_values = np.array(sim["feature_history"])
        v_values = np.array(sim["action_history"])
        # c_values = np.array(sim["combined_history"])
        trials = np.arange(1, q_values.shape[0] + 1)
        ax[0].scatter(trials, q_values[:, 0], label=r'$Q_0$', c='#FF00BF', alpha=0.5)
        ax[0].scatter(trials, q_values[:, 1], label=r'$Q_1$', c='#FF68A2', alpha=0.5)
        ax[0].scatter(trials, v_values[:, 0], label=r'$V_0$', c='#b2f7ef', alpha=0.5)
        ax[0].scatter(trials, v_values[:, 1], label=r'$V_1$', c='#5EB9FF', alpha=0.5)
        # ax[0].scatter(trials, c_values[:, 0], label=r'$C_0$', c='#D3D3D3', alpha=0.5)
        # ax[0].scatter(trials, c_values[:, 1], label=r'$C_1$', c='#3F3F3F', alpha=0.5)
        ax[0].set_xlabel('Trial')
        ax[0].set_ylabel('Value')
        # ax.set_ylim(-1.1, 1.1)
        ax[0].set_title(title)
        ax[0].legend(loc='upper left')
        feat_choices = np.array(sim["feature_choices"])
        act_choices = np.array(sim["action_choices"])
        rewards = np.array(sim["rewards"])
        fcma = moving_average(feat_choices, window)
        acma = moving_average(act_choices, window)
        rma = moving_average(rewards, window)

        ax[1].plot(trials, 1 - fcma, label=r'Faature choice = 0', c='#FFA3AC', lw=3)
        ax[1].plot(trials, fcma, label=r'Feature choice = 1', c='#B80000', lw=3)
        ax[1].plot(trials, 1 - acma, label=r'Action choice = 0', c='#5EB9FF', lw=3)
        ax[1].plot(trials, acma, label=r'Action choice = 1', c='#050C9C', lw=3)
        ax[1].plot(trials, rma, label='Reward', c='#00B800', lw=3)
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel('Trial')
        ax[1].set_ylabel('Prob.')
        ax[1].set_title(title)
        ax[1].legend(loc="upper left")

        fig.tight_layout()

        return fig
