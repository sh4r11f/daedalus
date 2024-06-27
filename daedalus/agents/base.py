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

    def __init__(self, name, root, n_actions, n_states, **kwargs):
        """
        Initialize the Q-learning agent with the learning rate (alpha), discount factor (gamma),
        number of states, number of actions, and the exploration rate (epsilon).
        """
        self.name = name
        self.root = root
        self.n_actions = n_actions
        self.n_states = n_states
        self._Q = np.zeros((self.n_states, self.n_actions))

        # Get/initialize some parameters
        self.n_params = 3
        self._alpha = kwargs.get("alpha", 0.5)
        self._beta = kwargs.get("beta", 0.5)
        self._bias = kwargs.get("bias", 0.5)
        self._init_val = kwargs.get("init_val", 0.5)

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

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value
