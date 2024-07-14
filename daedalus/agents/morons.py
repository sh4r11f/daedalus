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


class Vanilla(Agent):
    """
    RL agent with feature-based learning.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, **kwargs)

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values
        super().update(action, reward)


class VanillaDecay(Agent):
    """
    RL agent with feature-based learning.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, decay=0.5, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, **kwargs)

        self.decay = decay
        self._params.append(["decay", self.decay])
        self.bounds.append(kwargs.get("decay_bounds", ("decay", (1e-5, 1))))

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values
        super().update(action, reward)

        # Decay the unchosen action
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * self.kiyoo[1 - action]


class RewUnrew(Agent):
    """
    RL agent with feature-based learning.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        alpha_genuius (float): the genius learning rate for unchosen actions
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, alpha_unr=0.5, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, **kwargs)

        self.alpha_unr = alpha_unr
        self._params.append(["alpha_unr", self.alpha_unr])
        self.bounds.append(kwargs.get("alpha_unr_bounds", ("alpha_unr", (1e-5, 1))))

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values for rewarded actions
        if reward == 1:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha * (1 - self.kiyoo[action])
        # Update the Q-values for unrewarded actions
        else:
            self.kiyoo[action] = self.kiyoo[action] - self.alpha_unr * self.kiyoo[action]


class RewUnrewDecay(RewUnrew):
    """
    """
    def __init__(self, name, decay=0.5, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, **kwargs)

        self.decay = decay
        self._params.append(["decay", self.decay])
        self.bounds.append(kwargs.get("decay_bounds", ("decay", (1e-5, 1))))

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values for rewarded actions
        super().update(action, reward)

        # Decay the unchosen action
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * self.kiyoo[1 - action]


class HybridRewUnrew(RewUnrew):
    """
    """
    def __init__(self, name, omega=0.5, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, **kwargs)

        self.omega = omega
        self._params.append(["omega", self.omega])
        self.bounds.append(kwargs.get("omega_bounds", ("omega", (1e-5, 1))))

        # Initialize the other set of values
        self.vee = np.zeros(self.n_actions)
        self.vee_history = []

    def update(self, action, feature, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values
        if reward == 1:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha * (reward - self.kiyoo[action])
        else:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha_unr * (reward - self.kiyoo[action])

        # Update the feature values
        if reward == 1:
            self.vee[feature] = self.vee[feature] + self.alpha * (reward - self.vee[feature])
        else:
            self.vee[feature] = self.vee[feature] + self.alpha_unr * (reward - self.vee[feature])

    def reset(self):
        """
        Reset the agent.
        """
        super().reset()
        self.vee = np.zeros(self.n_actions)

    def choose_action(self, left_feat):
        """
        Chooses an action based on the current state.

        Returns:
            int: action chosen by the agent
        """
        probs = self.get_choice_probs(left_feat)
        return 0 if np.random.rand() < probs[0] else 1

    def get_choice_probs(self, left_feat):
        """
        Compute the choice probabilities.
        """
        probs = np.zeros(self.n_actions)

        # Compute the difference in Q-values
        act_diff = self.kiyoo[0] - self.kiyoo[1]
        feat_diff = self.vee[left_feat] - self.vee[1 - left_feat]

        # Compute the weighted logits
        q_diff = (1 - self.omega) * act_diff + self.omega * feat_diff

        # Compute the probabilities
        logits = (1 / self.sigma) * q_diff + self.bias
        probs[0]= self.sigmoid(logits)
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
        self.params = params

        # Run the agent on the data
        nll = 0
        for trial in data:

            # Unpack the trial
            action, reward, left_feat = trial

            # Get the feature choice
            feature = left_feat if action == 0 else 1 - left_feat  # left and right features are complementary

            # Update the Q-values
            self.update(action, feature, reward)

            # Compute the choice probabilities
            probs = self.get_choice_probs(left_feat)

            # Compute the log-likelihood
            log_like = np.log(probs[action] + 1e-10)
            self.hoods.append(log_like)
            nll -= log_like

        return nll


class HybridRewUnrewDecay(HybridRewUnrew):
    """
    """
    def __init__(self, name, decay=0.5, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, **kwargs)

        self.decay = decay
        self._params.append(["decay", self.decay])
        self.bounds.append(kwargs.get("decay_bounds", ("decay", (1e-5, 1))))

    def update(self, action, feature, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values and features
        super().update(action, feature, reward)

        # Decay the unchosen action and feature
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * self.kiyoo[1 - action]
        self.vee[1 - feature] = self.vee[1 - feature] - self.decay * self.vee[1 - feature]

    @property
    def params(self):
        pars = super().params
        return *pars, self.decay

    @params.setter
    def params(self, values):
        super().params = values[:-1]
        self.decay = values[-1]
