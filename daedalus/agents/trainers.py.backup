#!/usr/bin/env python
"""
created 3/8/2024

@author Sharif Saleki

Description:
"""
from abc import abstractmethod
import numpy as np
from scipy.optimize import minimize

from .simple import SimpleRL


class RLTrainer:
    """
    A simple Q-learning agent for discrete state and action spaces with a softmax action selection policy and
    update rule based on action/feature choice.

    Args:


    """
    def __init__(self, name, data, model_type, loss_type="nll"):
        """
        Initialize the Q-learning agent
        """
        self.name = name
        self.data = data
        self.model_type = model_type
        self.loss_type = loss_type

        # Initialize the model
        self.model = None

    def fit(self, opt_method="L-BFGS-B", bounds=None, seed=11):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.
        """
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Initialize parameters
        init_params = np.random.rand(self.model.n_params)

        # Reset model values and parameters
        self.model.reset()

        # Minimize the negative log likelihood
        res = minimize(
            self.loss_fun,
            init_params,
            method=opt_method,
            bounds=bounds,
        )

        return res

    @abstractmethod
    def loss_fun(self, params):
        """
        Calculates the negative log-likelihood of the observed choices given the model parameters.

        Args:
            params: list-like
                The model parameters: rewarded learning rate, unrewarded learning rate, beta, bias, and decay

        Returns:
            float: The negative log-likelihood of the observed choices given the model parameters.
        """
        pass


class SimpleQTrainer(RLTrainer):
    """
    A simple Q-learning agent for discrete state and action spaces with a softmax action selection policy and
    update rule based on action/feature choice.

    Args:
    """
    def __init__(self, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(**kwargs)
        self.n_states = kwargs.get("n_states", 2)
        self.n_actions = kwargs.get("n_actions", 2)

        self.model = SimpleRL(n_states=self.n_states, n_actions=self.n_actions, **kwargs)

    def loss_fun(self, params):
        """
        Calculates the negative log-likelihood of the observed choices given the model parameters.

        Args:
            params:

        Returns:

        """
        if self.loss_type == "nll":
            return self._nll(params)
        else:
            raise NotImplementedError(f"Loss function {self.loss_type} not implemented.")

    def _nll(self, params):
        """
        Calculates the negative log-likelihood of the observed choices given the model parameters.

        Args:
            params: list-like
                The model parameters: rewarded learning rate, unrewarded learning rate, beta, bias, and decay

        Returns:
            float: The negative log-likelihood of the observed choices given the model parameters.
        """
        # Set model parameters
        self.model.set_params(params)

        # Initialize negative log-likelihood
        nll = 0

        # Loop over the data
        for idx, row in self.data.iterrows():

            # Check if this is the first trial of a session and reset Q-values if so
            if row["TRIAL"].astype(int) == 1:
                self.model.reset_Q()

            # Convert to int
            action = row[self.model.choice_type].astype(int)
            reward = row["REWARD"].astype(int)

            # Update model values
            self.model.update(action, reward)

            # Calculate the choice probabilities using the softmax function
            probs = self.model.binary_softmax()

            # Update the negative log-likelihood
            nll -= np.log(probs[action] + self.epsilon)

        return nll


class ChoiceRewardQTrainer(RLTrainer):
    """

    """
    def __init__(self, model, data, method="L-BFGS-B", epsilon=1e-10, loss="nll"):
        """
        Initialize the Q-learning agent
        """
        super().__init__(model, data, method, epsilon, loss)
        self.bounds = [(0, 1), (0, 1), (-np.inf, np.inf), (-np.inf, np.inf), (0, 1), (0, 1)]

        if loss == "nll":
            self.loss_fun = self._nll

    def _nll(self, params):
        """
        Calculates the negative log-likelihood of the observed choices given the model parameters.

        Args:
            params: list-like
                The model parameters: rewarded learning rate, unrewarded learning rate, beta, bias, and decay

        Returns:
            float: The negative log-likelihood of the observed choices given the model parameters.
        """
        # Set model parameters
        self.model.set_params(params)

        # Initialize negative log-likelihood
        nll = 0

        # Loop over the data
        for idx, row in self.data.iterrows():

            # Check if this is the first trial of a session and reset Q-values if so
            if row["TRIAL"].astype(int) == 1:
                self.model.reset_R()
                self.model.reset_C()
                self.model.reset_Q()

            # Convert to int
            choice = row[self.model.choice_type].astype(int)
            reward = row["REWARD"].astype(int)

            # Update model values
            self.model.update(choice, reward)

            # Calculate the choice probabilities using the softmax function
            probs = self.model.binary_softmax()

            # Update the negative log-likelihood
            nll -= np.log(probs[choice] + self.epsilon)

        return nll


class StimulusActionQTrainer(RLTrainer):
    """

    """
    def __init__(self, model, data, method="L-BFGS-B", epsilon=1e-10, loss="nll"):
        """
        Initialize the Q-learning agent
        """
        super().__init__(model, data, method, epsilon, loss)
        self.bounds = [(0, 1), (0, 1), (-np.inf, np.inf), (-np.inf, np.inf), (0, 1), (0, 1)]

        if loss == "nll":
            self.loss_fun = self._nll

    def _nll(self, params):
        """
        Calculates the negative log-likelihood of the observed choices given the model parameters.

        Args:
            params: list-like
                The model parameters: rewarded learning rate, unrewarded learning rate, beta, bias, and decay

        Returns:
            float: The negative log-likelihood of the observed choices given the model parameters.
        """
        # Set model parameters
        self.model.set_params(params)

        # Initialize negative log-likelihood
        nll = 0

        # Loop over the data
        for idx, row in self.data.iterrows():

            # Check if this is the first trial of a session and reset Q-values if so
            if row["TRIAL"].astype(int) == 1:
                self.model.reset_A()
                self.model.reset_S()
                self.model.reset_Q()

            # Convert to int
            action_choice = row["RF_CHOICE"].astype(int)
            stim_choice = row["STIM_CHOICE"].astype(int)
            if action_choice == 0 and stim_choice == 0:
                choice = 0
            elif action_choice == 0 and stim_choice == 1:
                choice = 1
            elif action_choice == 1 and stim_choice == 0:
                choice = 2
            else:
                choice = 3
            reward = row["REWARD"].astype(int)

            # Update model values
            self.model.update(choice, reward)

            # Get available actions
            # Action indices are [0, 1, 2, 3] corresponding to [left down, left up, right down, right up]
            if row["STIM_OUT_DIR"].astype(int) == 0 and row["STIM_IN_DIR"].astype(int) == 0:  # left down and right down
                available_actions = [0, 2]
            elif row["STIM_OUT_DIR"].astype(int) == 0 and row["STIM_IN_DIR"].astype(int) == 1:  # left down and right up
                available_actions = [0, 3]
            elif row["STIM_OUT_DIR"].astype(int) == 1 and row["STIM_IN_DIR"].astype(int) == 0:  # left up and right down
                available_actions = [1, 2]
            else:  # left up and right up
                available_actions = [1, 3]

            # Calculate the choice probabilities using the softmax function
            probs = self.model.binary_softmax(available_options=available_actions)

            # Update the negative log-likelihood
            nll -= np.log(probs[choice] + self.epsilon)

        return nll

# Path: src/agents/trainers.py
