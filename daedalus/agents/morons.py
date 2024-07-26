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
#                      RULE: DAYW
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
        Update the Q-value to decay the unchosen option.

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
            self.kiyoo[action] = self.kiyoo[action] + self.alpha * (1 - self.kiyoo[action])
            self.vee[feature] = self.vee[feature] + self.alpha * (1 - self.vee[feature])
        else:
            self.kiyoo[action] = self.kiyoo[action] - self.alpha_unr * self.kiyoo[action]
            self.vee[feature] = self.vee[feature] - self.alpha_unr * self.vee[feature]

    def reset(self):
        """
        Reset the agent.
        """
        self.kiyoo = np.zeros(self.n_actions)
        self.vee = np.zeros(self.n_actions)
        self.choices = []
        self.rewards = []
        self.history = []
        self.hoods = []

    @property
    def V(self):
        return self.vee

    def left_option_value(self, left_feat):
        return (1 - self.omega) * self.kiyoo[0] + self.omega * self.vee[left_feat]

    def right_option_value(self, left_feat):
        return (1 - self.omega) * self.kiyoo[1] + self.omega * self.vee[1 - left_feat]

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
            action, feature, reward = trial

            # Get the feature on the left
            left_feat = feature if action == 0 else 1 - feature

            # Compute the choice probabilities
            probs = self.get_choice_probs(left_feat)

            # Compute the log-likelihood
            log_like = np.log(probs[action] + 1e-10)
            self.hoods.append(log_like)
            nll -= log_like

            # Update the Q-values
            self.update(action, feature, reward)

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


class HybridRewUnrewDecayPicky(HybridRewUnrew):
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

        # Decay the unchosen action and feature only if the chosen action was rewarded
        if reward == 1:
            self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * self.kiyoo[1 - action]
            self.vee[1 - feature] = self.vee[1 - feature] - self.decay * self.vee[1 - feature]


class HybridRewUnrewDecayForced(HybridRewUnrew):
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

        # Decay the unchosen action and feature toward 0.5
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * (self.kiyoo[1 - action] - 0.5)
        self.vee[1 - feature] = self.vee[1 - feature] - self.decay * (self.vee[1 - feature] - 0.5)


class HybridRewUnrewNomega(RewUnrew):
    """
    """
    def __init__(self, name, omega=0.5, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, **kwargs)

        # Have a constant omega=0.5 and don't add it to parameters
        self.omega = 0.5

        # Initialize the other set of values
        self.vee = np.zeros(self.n_actions)

    def update(self, action, feature, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the values
        if reward == 1:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha * (1 - self.kiyoo[action])
            self.vee[feature] = self.vee[feature] + self.alpha * (1 - self.vee[feature])
        else:
            self.kiyoo[action] = self.kiyoo[action] - self.alpha_unr * self.kiyoo[action]
            self.vee[feature] = self.vee[feature] - self.alpha_unr * self.vee[feature]

    def reset(self):
        """
        Reset the agent.
        """
        self.kiyoo = np.zeros(self.n_actions)
        self.vee = np.zeros(self.n_actions)
        self.choices = []
        self.rewards = []
        self.history = []
        self.hoods = []

    @property
    def V(self):
        return self.vee

    def left_option_value(self, left_feat):
        return (1 - self.omega) * self.kiyoo[0] + self.omega * self.vee[left_feat]

    def right_option_value(self, left_feat):
        return (1 - self.omega) * self.kiyoo[1] + self.omega * self.vee[1 - left_feat]

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
        self.params = params

        # Run the agent on the data
        nll = 0
        for trial in data:

            # Unpack the trial
            action, feature, reward = trial

            # Get the feature on the left
            left_feat = feature if action == 0 else 1 - feature

            # Compute the choice probabilities
            probs = self.get_choice_probs(left_feat)

            # Compute the log-likelihood
            log_like = np.log(probs[action] + 1e-10)
            self.hoods.append(log_like)
            nll -= log_like

            # Update the Q-values
            self.update(action, feature, reward)

        return nll


class HybridRewUnrewNomegaDecay(HybridRewUnrewNomega):
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
        # Update the values
        super().update(action, feature, reward)

        # Decay the unchosen action and feature
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * self.kiyoo[1 - action]
        self.vee[1 - feature] = self.vee[1 - feature] - self.decay * self.vee[1 - feature]


class HybridRewUnrewNomegaDecayPicky(HybridRewUnrewNomega):
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
        # Update the values
        super().update(action, feature, reward)

        # Decay the unchosen action and feature only if the chosen action was rewarded
        if reward == 1:
            self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * self.kiyoo[1 - action]
            self.vee[1 - feature] = self.vee[1 - feature] - self.decay * self.vee[1 - feature]


class HybridRewUnrewNomegaDecayForced(HybridRewUnrewNomega):
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
        # Update the values
        super().update(action, feature, reward)

        # Decay the unchosen action and feature toward 0.5
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * (self.kiyoo[1 - action] - 0.5)
        self.vee[1 - feature] = self.vee[1 - feature] - self.decay * (self.vee[1 - feature] - 0.5)
