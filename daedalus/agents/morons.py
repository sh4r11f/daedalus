#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                    SCRIPT: unlearners.py
#
#
#               DESCRIPTION: Agemts with decay rates
#
#
#                      RULE: 
#
#
#
#                   CREATOR: Sharif Saleki
#                      TIME: 07-11-2024-7810598105114117
#                     SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
import numpy as np

from .base import Agent


class Dumb(Agent):
    """
    RL agent with feature-based learning.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, alpha=0.5, sigma=1, bias=0.1, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, alpha=alpha, **kwargs)

        self.alpha = alpha
        self.sigma = sigma
        self.bias = bias
        self.bounds = kwargs.get("bounds", [(1e-5, 1), (1e-5, 1), (-np.inf, np.inf)])

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        self.kiyoo[action] += self.alpha * (1 - self.kiyoo[action])

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self.kiyoo.copy())

    @property
    def params(self):
        return self.alpha, self.sigma, self.bias

    @params.setter
    def params(self, values):
        self.alpha, self.sigma, self.bias = values


class Dumber(Dumb):
    """
    RL agent with feature-based learning.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        alpha_genuius (float): the genius learning rate for unchosen actions
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, alpha=0.5, alpha_genius=0.5, sigma=1, bias=0.1, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, alpha=alpha, sigma=sigma, bias=bias, **kwargs)

        self.alpha_genius = alpha_genius
        self.bounds = kwargs.get("bounds", [(1e-5, 1), (1e-5, 1), (1e-5, 1), (-np.inf, np.inf)])

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        if reward == 1:
            self.kiyoo[action] += self.alpha * (1 - self.kiyoo[action])
        else:
            self.kiyoo[action] -= self.alpha_genius * self.kiyoo[action]

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self.kiyoo.copy())

    @property
    def params(self):
        return self.alpha, self.alpha_genius, self.sigma, self.bias

    @params.setter
    def params(self, values):
        self.alpha, self.alpha_genius, self.sigma, self.bias = values


class DumbHybrid(Agent):
    """
    RL agent with feature-based learning.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, alpha=0.5, sigma=1, bias=0.1, omega=0.5, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, alpha=alpha, **kwargs)

        self.alpha = alpha
        self.sigma = sigma
        self.bias = bias
        self.omega = omega
        self.bounds = kwargs.get("bounds", [(1e-5, 1), (1e-5, 1), (-np.inf, np.inf), (1e-5, 1)])

        self.vee = np.zeros(self.kiyoo.shape)

    def update(self, action, feature, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        self.kiyoo[action] += self.alpha * (1 - self.kiyoo[action])
        self.vee[feature] += self.alpha * (1 - self.vee[feature])

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self.kiyoo.copy())

    @property
    def params(self):
        return self.alpha, self.sigma, self.bias, self.omega

    @params.setter
    def params(self, values):
        self.alpha, self.sigma, self.bias, self.omega = values

    def choose_action(self, feat_left):
        """
        Chooses an action based on the current state.

        Returns:
            int: action chosen by the agent
        """
        probs = self.get_choice_probs(feat_left)
        return 0 if np.random.rand() < probs[0] else 1

    def get_choice_probs(self, left_feat):
        """
        Compute the choice probabilities.
        """
        act_diff = self.kiyoo[0] - self.kiyoo[1]
        feat_diff = self.vee[left_feat] - self.vee[1 - left_feat]
        q_diff = (1 - self.omega) * act_diff + self.omega * feat_diff
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
            action, reward, feat_left = trial
            if action == 0:
                feat_choice = feat_left
            else:
                feat_choice = 1 - feat_left
            self.update(action, feat_choice, reward)
            probs = self.get_choice_probs(feat_left)
            ll = np.log(probs[action] + 1e-10)
            self.losses.append(ll)
            nll -= ll
        return nll
