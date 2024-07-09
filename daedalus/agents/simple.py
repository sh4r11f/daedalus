#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: simple.py
#
#
#          DESCRIPTION: Simple Q-learning agent for discrete state and action spaces with a softmax action selection policy and update rule based on action/feature choice.
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 06-26-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
import numpy as np

from .base import BaseRL


class SimpleQ(BaseRL):
    """
    A simple Q-learning agent for discrete state and action spaces with a softmax action selection policy and
    update rule based on action/feature choice.

    Args:
        alpha: learning rate
        beta: ?
        bias: bias

    """
    def __init__(self, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(n_actions=2, **kwargs)

        # Initialize the Q-table with number of states and actions
        self._Q = np.zeros(self.n_actions)

        # Initialize the learning rate for reward and unrewarded actions
        self.n_params = 5
        self._alpha_rew = 0
        self._alpha_unr = 0
        self._beta = 0
        self._bias = 0
        self._decay = 0

    def reset(self):
        self._Q = np.zeros(self.n_actions) + np.random.rand(self.n_actions)

    def update(self, choice, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            choice (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Chosen action
        self._Q[choice] = self._Q[choice] + self._alpha_rew * (reward - self._Q[choice])

        # Update unchosen action
        self._Q[1 - choice] = self._Q[1 - choice] - self._decay * self._Q[1 - choice]

    def choose_action(self, available_actions):
        """
        Chooses an action based on the current state.

        Args:
            available_actions (list): list of available actions

        Returns:
            int: action chosen by the agent
        """
        # Calculate the choice probabilities using the softmax function
        probs = self.sigmoid()

        if len(available_actions) > 1:
            # Choose an action based on the probabilities
            choice = np.random.choice(available_actions, p=probs)
        else:
            # Choose the only available stimulus
            choice = available_actions[0]

        return choice

    def sigmoid(self, clip_value=10):
        """
        Compute softmax values for each sets of scores in x.

        Args:
            available_options (iterable): indices of available options
            clip_value (int): value to clip the argument values to avoid overflow

        Returns:
            list: softmax values for each set of scores
        """
        # Calculate the argument values for all 4 options
        exponent = -self._beta * (self._Q[0] - self._Q[1]) + self._bias

        # Clip the value to avoid extreme values which can cause overflow.
        clipped_exp = np.clip(exponent, -clip_value, clip_value)
        left_prob = 1 / (1 + np.exp(clipped_exp))
        probs = [left_prob, 1 - left_prob]

        return probs

    def set_params(self, params):
        """
        Update model parameters.

        Args:
            params (list): list of parameters to update
        """
        assert len(params) == self._n_params, f"Expected {self._n_params} parameters, got {len(params)}."
        self._alpha_rew, self._alpha_unr, self._beta, self._bias, self._decay = params

    @property
    def params(self):
        return self._alpha_rew, self._alpha_unr, self._beta, self._bias, self._decay

    @property
    def Q(self):
        return self._Q

    @property
    def alpha_rew(self):
        return self._alpha_rew

    @alpha_rew.setter
    def alpha_rew(self, value):
        self._alpha_rew = value

    @property
    def alpha_unr(self):
        return self._alpha_unr

    @alpha_unr.setter
    def alpha_unr(self, value):
        self._alpha_unr = value

    @property
    def decay(self):
        return self._decay

    @decay.setter
    def decay(self, value):
        self._decay = value


class MultiChoiceQ(BaseRL):
    """
    Simple Q learning agent that can make multiple choices at once in a binary choice task.

    Args:
        alpha_rew (float): learning rate for rewarded actions
        alpha_unr (float): learning rate for unrewarded actions
        beta (float): beta parameter for softmax function
        bias (float): bias parameter for softmax function
        decay (float): decay parameter for unchosen action
    """
    def __init__(self, name, root, n_actions, n_states, **kwargs):
        super().__init__(name, root, n_actions, n_states, **kwargs)

        # Initialize the Q-table with number of states and actions
        self._Q = np.zeros(self.n_actions)

        # Initialize the learning rate for reward and unrewarded actions
        self.n_params = 5
        self._alpha_rew = 0
        self._alpha_unr = 0
        self._beta = 0
        self._bias = 0
        self._decay = 0

    def reset(self):
        self._Q = np.zeros(self.n_actions) + np.random.rand(self.n_actions)

    def update(self, choice, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            choice (int): action chosen by the agent
            reward (int): reward received from the environment
        """
        # Chosen action
        if reward:
            self._Q[choice] = self._Q[choice] + self._alpha_rew * (reward - self._Q[choice])
            self._Q[1 - choice] = self._Q[1 - choice] + self._alpha_unr * ((1 - reward) - self._Q[1 - choice])
        else:
            self._Q[choice] = self._Q[choice] + self._alpha_unr * (reward - self._Q[choice])
            self._Q[1 - choice] = self._Q[1 - choice] - (1 - self._alpha_unr) * self._Q[1 - choice]