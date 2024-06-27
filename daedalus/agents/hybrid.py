#!/usr/bin/env python
"""
created 3/6/2024

@author Sharif Saleki

Description:
"""
from typing import Union

import numpy as np

from . import QLearningAgent


class ChoiceRewardQ(QLearningAgent):
    """
    A simple Q-learning agent for discrete state and action spaces with a softmax action selection policy and
    update rule based on action/feature choice.

    Args:
        alpha: learning rate
        beta: ?
        bias: bias

    """
    def __init__(self, choice_type, **params):
        """
        Initialize the Q-learning agent
        """
        super().__init__(**params)

        self._model_name = "ChoiceRewardQ"
        self._choice_type = choice_type
        self._n_params = 6  # lr_rew, lr_unr, beta, bias, decay, omega

        self._R = np.zeros(self.num_actions) + self._init_val
        self._C = np.zeros(self.num_actions) + self._init_val

    def reset(self):
        self.reset_R()
        self.reset_C()

    def update(self, choice, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.
        """
        # Update reward value for chosen action
        if reward:
            self._R[choice] = self._R[choice] + self._lr_rew * (reward - self._Q[choice])
        else:
            self._R[choice] = self._R[choice] + self._lr_unr * (reward - self._Q[choice])

        # Update reward value for unchosen action
        self._R[1 - choice] = self._R[1 - choice] - self._decay * self._R[1 - choice]

        # Update choice value for chosen action
        self._C[choice] = self._C[choice] + self._lr_choice * (choice - self._C[choice])

        # Combine the choice and reward values
        self._Q[choice] = self._R[choice] + self._omega * self._C[choice]

    def choose_action(self, available_actions: Union[list, np.ndarray] = None):
        """
        Chooses an action based on the current state.

        Args:
            available_actions: list or array

        Returns:
            choice: int
        """
        # Calculate the choice probabilities using the softmax function
        probs = self.binary_softmax()

        if available_actions is not None and self.choice_type == "STIM_CHOICE":
            # Choose the only available stimulus
            choice = available_actions[0]
        else:
            # Choose an action based on the probabilities
            choice = np.random.choice(np.arange(self.num_actions), p=probs)

        return choice

    def set_params(self, params):
        """
        Update model parameters.
        Args:
            params:

        Returns:

        """
        assert len(params) == self._n_params, f"Expected {self._n_params} parameters, got {len(params)}."
        self._lr_rew, self._lr_unr, self._beta, self._bias, self._decay, self._omega = params

    def reset_R(self):
        self._R = np.zeros(self.num_actions) + self._init_val

    def reset_C(self):
        self._C = np.zeros(self.num_actions) + self._init_val

    @property
    def choice_type(self):
        return self._choice_type

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = value


class StimulusActionQ(QLearningAgent):
    """
    A Q-learning agent for discrete state and action spaces with a softmax action selection policy and hybrid
    feature + action update rule.
    """
    def __init__(self, **params):
        """
        Initialize the Q-learning agent
        """
        super().__init__(**params)

        self._model_name = "StimulusActionQ"
        self._n_params = 6  # lr_rew, lr_unr, beta, bias, decay, eta

        self._A = np.zeros(self.num_actions) + self._init_val
        self._S = np.zeros(self.num_actions) + self._init_val
        self._Q = np.zeros(self.num_actions * 2) + self._init_val

    def reset(self):
        self.reset_A()
        self.reset_S()
        self.reset_Q()

    def update(self, choice, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.
        """
        if choice == 0:
            action_choice = 0
            stim_choice = 0
        elif choice == 1:
            action_choice = 0
            stim_choice = 1
        elif choice == 2:
            action_choice = 1
            stim_choice = 0
        else:
            action_choice = 1
            stim_choice = 1

        # Updated rewarded values
        if reward:
            self._A[action_choice] = self._A[action_choice] + self._lr_rew * (reward - self._A[action_choice])
            self._S[stim_choice] = self._S[stim_choice] + self._lr_rew * (reward - self._S[stim_choice])
        else:
            self.A[action_choice] = self._A[action_choice] + self._lr_unr * (reward - self._A[action_choice])
            self._S[stim_choice] = self._S[stim_choice] + self._lr_unr * (reward - self._S[stim_choice])

        # Decay unrewarded values
        self._A[1 - action_choice] = self._A[1 - action_choice] - self._decay * self._A[1 - action_choice]
        self._S[1 - stim_choice] = self._S[1 - stim_choice] - self._decay * self._S[1 - stim_choice]

        # Combine the choice and reward values
        self._Q[choice] = self._eta * self._A[action_choice] + (1 - self._eta) * self._S[stim_choice]

    def choose_action(self, available_options: Union[list, np.ndarray] = None):
        """
        Chooses an action based on the current state.
        """
        # Calculate the choice probabilities using the softmax function
        if available_options is not None:
            probs = self.binary_softmax(available_options)
        else:
            raise ValueError("No available options.")

        choice = np.random.choice(available_options, p=probs)

        return choice

    def set_params(self, params):
        """
        Update model parameters.
        Args:
            params:

        Returns:

        """
        assert len(params) == self._n_params, f"Expected {self._n_params} parameters, got {len(params)}."
        self._lr_rew, self._lr_unr, self._beta, self._bias, self._decay, self._eta = params

    def reset_A(self):
        self._A = np.zeros(self.num_actions) + self._init_val

    def reset_S(self):
        self._S = np.zeros(self.num_actions) + self._init_val

    def reset_Q(self):
        self._Q = np.zeros(self.num_actions * 2) + self._init_val

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, value):
        self._S = value

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value



