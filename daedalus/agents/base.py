#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                    SCRIPT: base.py
#
#
#               DESCRIPTION: RLs with feature-based learning
#
#
#                      RULE: DAYW
#
#
#
#                   CREATOR: Sharif Saleki
#                         TIME: 06-07-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
import numpy as np


class BaseGent:
    """
    Args:
        name (str): name of the agent
        alpha (float): learning rate
        n_actions (int): number of actions
    """
    def __init__(self, name, n_actions, n_states, alpha=0.5, sigma=1, bias=0.1, **kwargs):
        """
        Initialize the Q-learning agent with the learning rate (alpha), discount factor (gamma),
        number of states, number of actions, and the exploration rate (epsilon).
        """
        self.name = name
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.sigma = sigma
        self.bias = bias

        # Set the bounds
        self._params = [
            ["alpha", self.alpha],
            ["sigma", self.sigma],
            ["bias", self.bias],
            ]
        self.bounds = [
            kwargs.get("alpha_bounds", ("alpha", (1e-5, 1))),
            kwargs.get("sigma_bounds", ("sigma", (1e-5, 1))),
            kwargs.get("bias_bounds", ("bias", (-1e10, 1e10))),
            ]

        # Get/initialize parameters
        # self.kiyoo = np.zeros((self.n_states, self.n_actions))
        self.kiyoo = np.zeros(self.n_actions)
        self.init_val = kwargs.get("init_val", 0.5)
        self.clip_value = kwargs.get("clip_value", 1e8)
        self.version = kwargs.get("version", "0.0")

        # For saving history
        self.choices = []
        self.rewards = []
        self.history = []
        self.hoods = []

    @property
    def params(self):
        return tuple([param[1] for param in self._params])

    @params.setter
    def params(self, values):
        n_vals = len(values)
        n_params = len(self._params)
        if n_vals != n_params:
            err = f"Number of parameters {n_params} does not match the number of values {n_vals}."
            err += f"\nParameters: {self._params}\nValues: {values}"
            raise ValueError(err)

        for i, value in enumerate(values):
            if self._params[i][0] not in ["sigma", "bias"]:
                value = 1 / (1 + np.exp(-value))
            # value = np.clip(value, float(self.bounds[i][1][0]), float(self.bounds[i][1][1]))
            self._params[i][1] = value
            setattr(self, self._params[i][0], value)

    @property
    def Q(self):
        return self.kiyoo

    def reset(self):
        """
        Reset the Q-values to their initial values.
        """
        # self.kiyoo = np.zeros((self.n_states, self.n_actions))
        self.kiyoo = np.zeros(self.n_actions)
        self.choices = []
        self.rewards = []
        self.history = []
        self.hoods = []

    def sigmoid(self, x):
        """
        Compute softmax values for each sets of scores in x.

        Args:
            clip_value (int): value to clip the argument values to avoid overflow

        Returns:
            list: softmax values for each set of scores
        """
        x = np.clip(x, -self.clip_value, self.clip_value)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(q_values):
        """
        Compute softmax values for each sets of scores in x.

        Args:
            q_values (array): array of Q-values

        Returns:
            array: softmax values for each set of scores
        """
        exp_values = np.exp(q_values - np.max(q_values))  # stability improvement
        return exp_values / np.sum(exp_values)


class Agent(BaseGent):
    """
    Binary choice agent.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        n_actions (int): number of actions
    """
    def __init__(self, name, **kwargs):
        """
        Initialize the Q-learning agent with the learning rate (alpha), discount factor (gamma),
        number of states, number of actions, and the exploration rate (epsilon).
        """
        # Initialize the agent
        super().__init__(name, n_actions=2, n_states=1, **kwargs)
        self.kiyoo = np.zeros(self.n_actions)

    def reset(self):
        """
        Reset the Q-values to their initial values.
        """
        # self.kiyoo = np.zeros((self.n_states, self.n_actions))
        self.kiyoo = np.zeros(self.n_actions)
        self.choices = []
        self.rewards = []
        self.history = []
        self.hoods = []

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values
        self.kiyoo[action] = self.kiyoo[action] + self.alpha * (reward - self.kiyoo[action])

    def choose_action(self):
        """
        Chooses an action based on the current state.

        Returns:
            int: action chosen by the agent
        """
        probs = self.get_choice_probs()
        return 0 if np.random.rand() < probs[0] else 1

    def get_choice_probs(self):
        """
        Compute the choice probabilities.
        """
        # Initialize the probabilities
        probs = np.zeros_like(self.kiyoo)

        # Compute the difference in Q-values
        q_diff = self.kiyoo[0] - self.kiyoo[1]

        # Compute the probabilities
        logits = (1 / self.sigma) * q_diff + self.bias

        # Compute the probabilities
        probs[0] = self.sigmoid(logits)
        probs[1] = 1 - probs[0]

        return probs

    def loss(self, params, data):
        """
        Compute the loss of the agent.

        Returns:
            float: loss of the agent
        """
        # Reset the agent
        self.reset()

        # Update the parameters
        self.params = params

        # Run the agent on the data
        nll = 0
        for trial in data:

            # Unpack the trial
            action, _, reward = trial
            action, reward = int(action), int(reward)

            # Compute the choice probabilities
            probs = self.get_choice_probs()

            # Compute the log-likelihood
            log_like = np.log(probs[action] + 1e-10)
            self.hoods.append(log_like)

            # Update the log-likelihood
            nll -= log_like

            # Update the Q-values
            self.update(action, reward)

        return nll
