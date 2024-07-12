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
import json

import numpy as np

from .base import BaseRL


class SimpleQ(BaseRL):
    """
    A simple agent for discrete state and action spaces with a sigmoid action selection and
    RW learning rule.

    Args:
        alpha: learning rate
        beta: ?
        bias: bias

    """
    def __init__(self, name, alpha=0.5, decay=0.9, beta=1, bias=0.1, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, n_actions=2, n_states=1, **kwargs)

        # Initialize the learning rate for reward and unrewarded actions
        self._n_params = 3
        self._alpha = alpha
        self._decay = decay
        self._beta = beta
        self.bounds = kwargs.get("bounds", [(1e-5, 1), (1e-5, 1), (1, 10)])

        self.bias = bias

    def reset(self):
        self._Q = np.zeros(2)
        self.choices = []
        self.rewards = []
        self.history = []

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Chosen action
        self._Q[action] += self._alpha * (reward - self._Q[action])

        # Update unchosen action
        self._Q[1 - action] *= (1 - self._decay)

        # Clipping
        # self.Q = np.clip(self.Q, -self.clip_value, self.clip_value)  # Clip values to avoid overflow

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self._Q.copy())

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
        q_diff = self._Q[0] - self._Q[1]
        logits = self._beta * q_diff + self.bias
        prob_left = self.sigmoid(logits)
        return [prob_left, 1 - prob_left]

    def loss(self, params, data):
        """
        Compute the loss of the agent.

        Returns:
            float: loss of the agent
        """
        self.set_params(params)
        self.reset()
        nll = 0
        for trial in data:
            action, reward = trial
            self.update(action, reward)
            probs = self.get_choice_probs()
            nll -= np.log(probs[action] + 1e-10)
        return nll

    def set_params(self, params):
        """
        Update model parameters.

        Args:
            params (list): list of parameters to update
        """
        self._alpha, self._decay, self._beta = params

    def save(self, filename=None):
        """
        Save the agent's state to a file.
        """
        path = self.root / "data" / "RL"
        path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{self.name}_state_v{self.version}.json"
        file = path / filename

        state = {
            'Q': self._Q.tolist(),
            'alpha': self._alpha,
            'decay': self._decay,
            'beta': self._beta,
            'bias': self.bias,
            'choices': self.choices,
            'rewards': self.rewards,
            'history': [q.tolist() for q in self.history]
        }
        with open(file, 'w') as f:
            json.dump(state, f)

    def load(self, filename=None):
        """
        Load the agent's state from a file.
        """
        path = self.root / "data" / "RL"
        path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{self.name}_state_v{self.version}.json"
        file = path / filename
        if file.is_file():
            with open(filename, 'r') as f:
                state = json.load(f)
            self._Q = np.array(state['Q'])
            self._alpha = state['alpha']
            self._decay = state['decay']
            self._beta = state['beta']
            self.bias = state['bias']
            self.choices = state['choices']
            self.rewards = state['rewards']
            self.history = [np.array(q) for q in state['history']]
        else:
            print(f"File {filename} not found. Agent state not loaded.")

    @property
    def params(self):
        return self._alpha, self._decay, self._beta

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def decay(self):
        return self._decay

    @decay.setter
    def decay(self, value):
        self._decay = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
