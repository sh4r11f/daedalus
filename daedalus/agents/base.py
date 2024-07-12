#!/usr/bin/env python
"""
created 3/6/2024

@author Sharif Saleki

Description:
"""
from typing import Union
from abc import abstractmethod

import numpy as np


class BaseRL:
    """
    A Q-learning agent for discrete state and action spaces.

    Args:
        name (str): name of the agent
        root (str): root directory for the agent
        n_actions (int): number of actions
        n_states (int): number of states
    """

    def __init__(self, name, root, version, n_actions, n_states, **kwargs):
        """
        Initialize the Q-learning agent with the learning rate (alpha), discount factor (gamma),
        number of states, number of actions, and the exploration rate (epsilon).
        """
        self.name = name
        self.root = root
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

    @abstractmethod
    def reset(self):
        pass

    # Method to be implemented
    @abstractmethod
    def update(self, choice, reward):
        pass

    @abstractmethod
    def choose_action(self, available_actions: Union[list, np.ndarray]):
        """
        Chooses an action based on the current state.

        Args:
            available_actions (list, np.ndarray): list of available actions
        """
        pass

    @property
    def n_params(self):
        return self._n_params

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value
