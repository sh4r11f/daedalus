#!/usr/bin/env python
"""
created 3/6/2024

@author Sharif Saleki

Description:
"""
import numpy as np


class BaseRL:
    """
    A Q-learning agent for discrete state and action spaces.

    Args:
        name (str): name of the agent
        root (str): root directory for the agent
        version (str): version of the agent
        n_actions (int): number of actions
        n_states (int): number of states
    """

    def __init__(self, name, n_actions, n_states, version, **kwargs):
        """
        Initialize the Q-learning agent with the learning rate (alpha), discount factor (gamma),
        number of states, number of actions, and the exploration rate (epsilon).
        """
        self.name = name
        self.version = version
        self.n_actions = n_actions
        self.n_states = n_states
        self._Q = np.zeros((self.n_states, self.n_actions))

        # Get/initialize some parameters
        self._n_params = 0
        self._init_val = kwargs.get("init_val", 0.5)
        self.clip_value = kwargs.get("clip_value", 100)

        # Saving history
        self.choices = []
        self.rewards = []
        self.history = []

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

    @property
    def n_params(self):
        return self._n_params

    @property
    def Q(self):
        return self._Q


class Agent:
    """
    A Q-learning agent for discrete state and action spaces.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        n_actions (int): number of actions
    """

    def __init__(self, name, alpha=0.5, n_actions=2, **kwargs):
        """
        Initialize the Q-learning agent with the learning rate (alpha), discount factor (gamma),
        number of states, number of actions, and the exploration rate (epsilon).
        """
        self.name = name
        self.alpha = alpha
        self.n_actions = n_actions

        # Get/initialize parameters
        self.kiyoo = np.zeros(self.n_actions)

        self.init_val = kwargs.get("init_val", 0.5)
        self.clip_value = kwargs.get("clip_value", 1e8)
        self.version = kwargs.get("version", "0.0")

        # For saving history
        self.choices = []
        self.rewards = []
        self.history = []
        self.losses = []

    def reset(self):
        """
        Reset the Q-values to their initial values.
        """
        self.kiyoo = np.zeros(self.n_actions)
        self.choices = []
        self.rewards = []
        self.history = []
        self.losses = []

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
        q_diff = self.kiyoo[0] - self.kiyoo[1]
        logits = (1 / self.sigma) * q_diff + self.bias
        prob_left = self.sigmoid(logits)
        return [prob_left, 1 - prob_left]

    def loss(self, params, data):
        """
        Compute the loss of the agent.

        Returns:
            float: loss of the agent
        """
        self.params = params
        self.reset()
        nll = 0
        for trial in data:
            action, reward = trial
            self.update(action, reward)
            probs = self.get_choice_probs()
            ll = np.log(probs[action] + 1e-10)
            self.losses.append(ll)
            nll -= ll
        return nll

    @property
    def params(self):
        return self.alpha

    @params.setter
    def params(self, *args):
        self.alpha = args[0]

    @property
    def Q(self):
        return self.kiyoo

