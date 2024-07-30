#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: zoo.py
#
#
#                   DESCRIPTION: Model zoo
#
#
#                          RULE: Import and use
#
#
#
#                       CREATOR: Sharif Saleki
#                          TIME: 07-29-2024-7810598105114117
#                         SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
import numpy as np
import random

from .base import BaseGent, Agent


class ActionBased(Agent):
    """
    RL agent with action-based learning.
    """
    def __init__(self, name, alpha_unr=0.5, **kwargs):
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


class ActionBasedDecay(ActionBased):
    """
    Action-based learning with decay for unchosen actions.
    """
    def __init__(self, name, decay=0.5, **kwargs):
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
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * (self.kiyoo[1 - action] - 0.5)


class ActionBasedCoupled(ActionBased):
    """
    Action-based learning with coupled learning rates.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

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
            self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.alpha * self.kiyoo[1 - action]
        # Update the Q-values for unrewarded actions
        else:
            self.kiyoo[action] = self.kiyoo[action] - self.alpha_unr * self.kiyoo[action]
            self.kiyoo[1 - action] = self.kiyoo[1 - action] + self.alpha_unr * (1 - self.kiyoo[1 - action])


class FeatureBased(Agent):
    """
    RL agent with feature-based learning.
    """
    def __init__(self, name, alpha_unr=0.5, **kwargs):
        super().__init__(name, **kwargs)

        self.vee = np.zeros(2)
        self.alpha_unr = alpha_unr
        self._params.append(["alpha_unr", self.alpha_unr])
        self.bounds.append(kwargs.get("alpha_unr_bounds", ("alpha_unr", (1e-5, 1))))

    @property
    def V(self):
        """
        Return the value of the features.
        """
        return self.vee

    def reset(self):
        """
        Reset the Q-values to their initial values.
        """
        self.vee = np.zeros(2)
        self.choices = []
        self.rewards = []
        self.history = []
        self.hoods = []

    def update(self, feature, reward):
        """
        Update the value for a given feature and reward.

        Args:
            feature (int): feature chosen by the agent
            reward (float): reward received from the environment
        """
        if reward == 1:
            self.vee[feature] = self.vee[feature] + self.alpha * (1 - self.vee[feature])
        else:
            self.vee[feature] = self.vee[feature] - self.alpha_unr * self.vee[feature]

    def get_choice_probs(self):
        """
        Compute the choice probabilities.
        """
        # Initialize the probabilities
        probs = np.zeros_like(self.vee)

        # Compute the difference in values
        diff = self.vee[0] - self.vee[1]

        # Compute the probabilities
        logits = (1 / self.sigma) * diff + self.bias

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
            _, feature, reward = trial
            feature, reward = int(feature), int(reward)

            # Compute the choice probabilities
            probs = self.get_choice_probs()

            # Compute the log-likelihood
            log_like = np.log(probs[feature] + 1e-10)
            self.hoods.append(log_like)

            # Update the log-likelihood
            nll -= log_like

            # Update the Q-values
            self.update(feature, reward)

        return nll


class FeatureBasedDecay(FeatureBased):
    """
    Feature-based learning with decay for unchosen features.
    """
    def __init__(self, name, decay=0.5, **kwargs):
        super().__init__(name, **kwargs)

        self.decay = decay
        self._params.append(["decay", self.decay])
        self.bounds.append(kwargs.get("decay_bounds", ("decay", (1e-5, 1))))

    def update(self, feature, reward):
        """
        Update the value for a given feature and reward.

        Args:
            feature (int): feature chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values for rewarded actions
        super().update(feature, reward)

        # Decay the unchosen action
        self.vee[1 - feature] = self.vee[1 - feature] - self.decay * (self.vee[1 - feature] - 0.5)


class FeatureBasedCoupled(FeatureBased):
    """
    Feature-based learning with coupled learning rates.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def update(self, feature, reward):
        """
        Update the value for a given feature and reward.

        Args:
            feature (int): feature chosen by the agent
            reward (float): reward received from the environment
        """
        if reward == 1:
            self.vee[feature] = self.vee[feature] + self.alpha * (1 - self.vee[feature])
            self.vee[1 - feature] = self.vee[1 - feature] - self.alpha * self.vee[1 - feature]
        else:
            self.vee[feature] = self.vee[feature] - self.alpha_unr * self.vee[feature]
            self.vee[1 - feature] = self.vee[1 - feature] + self.alpha_unr * (1 - self.vee[1 - feature])


class Hybrid(ActionBased):
    """
    RL agent with hybrid learning.
    """
    def __init__(self, name, omega=0.5, **kwargs):
        super().__init__(name, **kwargs)

        self.vee = np.zeros(2)
        self.omega = omega
        self._params.append(["omega", self.omega])
        self.bounds.append(kwargs.get("omega_bounds", ("omega", (1e-5, 1))))

    @property
    def V(self):
        """
        Return the value of the features.
        """
        return self.vee

    def reset(self):
        """
        Reset the Q-values to their initial values.
        """
        self.kiyoo = np.zeros(2)
        self.vee = np.zeros(2)
        self.choices = []
        self.rewards = []
        self.history = []
        self.hoods = []

    def update(self, action, feature, reward):
        """
        Update the value for a given feature and reward.

        Args:
            action (int): action chosen by the agent
            feature (int): feature chosen by the agent
            reward (float): reward received from the environment
        """
        if reward == 1:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha * (1 - self.kiyoo[action])
            self.vee[feature] = self.vee[feature] + self.alpha * (1 - self.vee[feature])
        else:
            self.kiyoo[action] = self.kiyoo[action] - self.alpha_unr * self.kiyoo[action]
            self.vee[feature] = self.vee[feature] - self.alpha_unr * self.vee[feature]

    def left_option_value(self, left_feat):
        return (1 - self.omega) * self.kiyoo[0] + self.omega * self.vee[left_feat]

    def right_option_value(self, left_feat):
        return (1 - self.omega) * self.kiyoo[1] + self.omega * self.vee[1 - left_feat]

    def choose(self, left_feat):
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
        probs = np.zeros(2)

        # Compute the difference in Q-values
        act_diff = self.kiyoo[0] - self.kiyoo[1]
        feat_diff = self.vee[left_feat] - self.vee[1 - left_feat]

        # Compute the weighted logits
        diff = (1 - self.omega) * act_diff + self.omega * feat_diff

        # Compute the probabilities
        logits = (1 / self.sigma) * diff + self.bias
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
            action, feature, reward = int(action), int(feature), int(reward)

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


class HybridDecay(Hybrid):
    """
    Hybrid learning with decay for unchosen actions and features.
    """
    def __init__(self, name, decay=0.5, **kwargs):
        super().__init__(name, **kwargs)

        self.decay = decay
        self._params.append(["decay", self.decay])
        self.bounds.append(kwargs.get("decay_bounds", ("decay", (1e-5, 1))))

    def update(self, action, feature, reward):
        """
        Update the value for a given feature and reward.

        Args:
            action (int): action chosen by the agent
            feature (int): feature chosen by the agent
            reward (float): reward received from the environment
        """
        # Update the Q-values for rewarded actions
        super().update(action, feature, reward)

        # Decay the unchosen action
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * (self.kiyoo[1 - action] - 0.5)

        # Decay the unchosen feature
        self.vee[1 - feature] = self.vee[1 - feature] - self.decay * (self.vee[1 - feature] - 0.5)


class HybridCoupled(Hybrid):
    """
    Hybrid learning with coupled update for chosen and unchosen.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def update(self, action, feature, reward):
        """
        Update the value for a given feature and reward.

        Args:
            action (int): action chosen by the agent
            feature (int): feature chosen by the agent
            reward (float): reward received from the environment
        """
        if reward == 1:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha * (1 - self.kiyoo[action])
            self.vee[feature] = self.vee[feature] + self.alpha * (1 - self.vee[feature])
            self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.alpha * self.kiyoo[1 - action]
            self.vee[1 - feature] = self.vee[1 - feature] - self.alpha * self.vee[1 - feature]
        else:
            self.kiyoo[action] = self.kiyoo[action] - self.alpha_unr * self.kiyoo[action]
            self.vee[feature] = self.vee[feature] - self.alpha_unr * self.vee[feature]
            self.kiyoo[1 - action] = self.kiyoo[1 - action] + self.alpha_unr * (1 - self.kiyoo[1 - action])
            self.vee[1 - feature] = self.vee[1 - feature] + self.alpha_unr * (1 - self.vee[1 - feature])


class ObjectBased(BaseGent):
    def __init__(self, name, alpha_unr=0.5, **kwargs):
        super().__init__(name, n_actions=4, n_states=1, **kwargs)

        self.kiyoo = np.zeros(self.n_actions)
        self.alpha_unr = alpha_unr
        self._params.append(["alpha_unr", self.alpha_unr])
        self.bounds.append(
            kwargs.get("alpha_unr_bounds", ("alpha_unr", (1e-5, 1)))
            )

    def reset(self):
        self.kiyoo = np.zeros(self.n_actions)
        self.choices = []
        self.rewards = []
        self.history = []
        self.hoods = []

    def update(self, obj, reward):
        """
        Update object values based on the choice and reward received.
        """
        if reward == 1:
            self.kiyoo[obj] = self.kiyoo[obj] + self.alpha * (1 - self.kiyoo[obj])
        else:
            self.kiyoo[obj] = self.kiyoo[obj] - self.alpha_unr * self.kiyoo[obj]

    def choose_action(self, options):
        probs = self.get_choice_probs(options)
        return options[0] if random.random() < probs[0] else options[1]

    def get_choice_probs(self, options):

        # Initialize probabilities
        probs = np.zeros(2)

        # Calculate difference in values
        diff = self.kiyoo[options[0]] - self.kiyoo[options[1]]

        # Calculate probabilities
        logits = (1 / self.sigma) * diff + self.bias
        probs[0] = self.sigmoid(logits)
        probs[1] = 1 - probs[0]

        return probs

    def loss(self, params, data):

        # Reset
        self.reset()
        self.params = params

        # Run on data
        nll = 0
        for trial in data:

            # Unpack
            action, feature, reward = trial
            action, feature, reward = int(action), int(feature), int(reward)

            # Find object number
            # chosen = action * 2 + feature
            # unchosen = 3 - chosen
            # options = [chosen, unchosen] if action == 0 else [unchosen, chosen]
            if action == 0:
                if feature == 0:
                    options = [0, 3]
                    choice = 0
                else:
                    options = [1, 2]
                    choice = 1
            else:
                if feature == 0:
                    options = [1, 2]
                    choice = 2
                else:
                    options = [0, 3]
                    choice = 3

            # Calculate the loss
            probs = self.get_choice_probs(options)
            log_like = np.log(probs[action] + 1e-10)
            self.hoods.append(log_like)

            nll -= log_like

            # Update the value
            self.update(choice, reward)

        return nll


class ObjectBasedDecay(ObjectBased):
    def __init__(self, name, decay=0.5, **kwargs):
        super().__init__(name, **kwargs)

        self.decay = decay
        self._params.append(["decay", self.decay])
        self.bounds.append(
            kwargs.get("decay_bounds", ("decay", (1e-5, 1)))
            )

    def update(self, obj, reward):
        """
        Update object values based on the choice and reward received.
        Decay the value of all other objects.
        """
        # Update the Q-values for rewarded actions
        super().update(obj, reward)

        # Decay the unchosen objects
        for unchosen in range(4):
            if unchosen != obj:
                self.kiyoo[unchosen] = self.kiyoo[unchosen] - self.decay * (self.kiyoo[unchosen] - 0.5)


class ObjectBasedCoupled(ObjectBased):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def update(self, obj, reward):
        """
        Update object values based on the choice and reward received.
        Update the value of all other objects.
        """
        if reward == 1:
            self.kiyoo[obj] = self.kiyoo[obj] + self.alpha * (1 - self.kiyoo[obj])
            self.kiyoo[3 - obj] = self.kiyoo[3 - obj] - self.alpha * self.kiyoo[3 - obj]
        else:
            self.kiyoo[obj] = self.kiyoo[obj] - self.alpha_unr * self.kiyoo[obj]
            self.kiyoo[3 - obj] = self.kiyoo[3 - obj] + self.alpha_unr * (1 - self.kiyoo[3 - obj])
