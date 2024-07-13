#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: smart.py
#
#
#               DESCRIPTION: Agents that are so smart they learn if they don't get rewarded
#
#
#                          RULE: DAYW
#
#
#
#                   CREATOR: Sharif Saleki
#                         TIME: 07-11-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
import json

import numpy as np

from .base import Agent


class AgentLeman(Agent):
    """
    RL agent with a decay parameter.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        decay (float): decay rate
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, alpha=0.5, decay=0.9, sigma=1, bias=0.1, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, alpha=alpha, **kwargs)

        self.decay = decay
        self.sigma = sigma
        self.bias = bias
        self.bounds = kwargs.get("bounds", [
            (1e-5, 1),  # alpha
            (1e-5, 1),  # decay
            (1, 10),  # sigma
            (-np.inf, np.inf)  # bias
            ])

    @property
    def params(self):
        return self.alpha, self.decay, self.sigma, self.bias

    @params.setter
    def params(self, values):
        self.alpha, self.decay, self.sigma, self.bias = values


    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        # Update chosen action
        self.kiyoo[action] = self.kiyoo[action] + self.alpha * (reward - self.kiyoo[action])

        # Decay unchosen action
        self.kiyoo[1 - action] = self.kiyoo[1 - action] - self.decay * self.kiyoo[1 - action]

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self.kiyoo.copy())

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
            'Q': self.kiyoo.tolist(),
            'alpha': self.alpha,
            'decay': self.decay,
            'sigma': self.sigma,
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
            self.kiyoo = np.array(state['Q'])
            self.alpha = state['alpha']
            self.decay = state['decay']
            self.sigma = state['sigma']
            self.bias = state['bias']
            self.choices = state['choices']
            self.rewards = state['rewards']
            self.history = [np.array(q) for q in state['history']]
        else:
            print(f"File {filename} not found. Agent state not loaded.")


class DecayGent(AgentLeman):
    """
    RL agent with a decay parameter.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        decay (float): decay rate
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, alpha=0.5, alpha_genius=0.5, decay=0.9, sigma=1, bias=0.1, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, alpha=alpha, decay=decay, sigma=sigma, bias=bias, **kwargs)

        self.alpha_genius = alpha_genius
        self.bounds = kwargs.get("bounds", [
            (1e-5, 1),  # alpha
            (1e-5, 1),  # alpha_genius
            (1e-5, 1),  # decay
            (1, 10),  # sigma
            (-np.inf, np.inf)  # bias
            ])

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

        # Decay unchosen action
        self.kiyoo[1 - action] -= self.decay * self.kiyoo[1 - action]

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self.kiyoo.copy())

    @property
    def params(self):
        return self.alpha, self.alpha_genius, self.decay, self.sigma, self.bias

    @params.setter
    def params(self, values):
        self.alpha, self.alpha_genius, self.decay, self.sigma, self.bias = values


class HyberGent(AgentLeman):
    """
    RL agent with feature-based learning.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, alpha=0.5, decay=0.8, sigma=1, bias=0.1, omega=0.5, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, alpha=alpha, decay=decay, sigma=sigma, bias=bias, **kwargs)

        self.omega = omega
        self.bounds = kwargs.get("bounds", [(1e-5, 1), (1e-5, 1), (-np.inf, np.inf), (1e-5, 1), (1e-5, 1)])

        self.vee = np.zeros(self.kiyoo.shape)
        self.vee_history = []

    def update(self, action, feature, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        self.kiyoo[action] += self.alpha * (1 - self.kiyoo[action])
        self.kiyoo[1 - action] -= self.decay * self.kiyoo[1 - action]

        self.vee[feature] += self.alpha * (1 - self.vee[feature])
        self.vee[1 - feature] -= self.decay * self.vee[1 - feature]

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self.kiyoo.copy())
        self.vee_history.append(self.vee.copy())

    @property
    def params(self):
        return self.alpha, self.decay, self.sigma, self.bias, self.omega

    @params.setter
    def params(self, values):
        self.alpha, self.decay, self.sigma, self.bias, self.omega = values

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


class ObJent(AgentLeman):
    """
    RL agent with feature-based learning.

    Args:
        name (str): name of the agent
        alpha (float): learning rate
        sigma (float): standard deviation of the Gaussian noise
        bias (float): bias
    """
    def __init__(self, name, alpha=0.5, decay=0.8, sigma=1, bias=0.1, **kwargs):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, n_actions=4, alpha=alpha, decay=decay, sigma=sigma, bias=bias, **kwargs)

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        for i in range(self.n_actions):
            if i == action:
                self.kiyoo[i] += self.alpha * (reward - self.kiyoo[i])
            else:
                self.kiyoo[i] -= self.decay * self.kiyoo[i]

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self.kiyoo.copy())

    def choose_action(self, options):
        """
        Chooses an action based on the current state.

        Returns:
            int: action chosen by the agent
        """
        probs = self.get_choice_probs(options)
        return options[0] if np.random.rand() < probs[options[0]] else options[1]

    def get_choice_probs(self, options):
        """
        Compute the choice probabilities.
        """
        probs = np.zeros(self.n_actions)
        q_diff = self.kiyoo[options[0]] - self.kiyoo[options[1]]
        logits = (1 / self.sigma) * q_diff + self.bias
        probs[options] = self.sigmoid(logits)
        return probs

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
            action, reward, mot_left = trial
            if mot_left == 0:
                options = [0, 3]
                if action == 0:
                    slc = 0
                else:
                    slc = 3
            else:
                options = [1, 2]
                if action == 0:
                    slc = 1
                else:
                    slc = 2
            self.update(slc, reward)
            probs = self.get_choice_probs(options)
            ll = np.log(probs[action] + 1e-10)
            self.losses.append(ll)
            nll -= np.log(probs[action] + 1e-10)
        return nll

    def save(self, filename=None):
        """
        Save the agent's state to a file.
        """
        path = self.root / "data" / "RL"
        path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{self.name}_v{self.version}.json"
        file = path / filename

        state = {
            'Q': self.kiyoo.tolist(),
            'alpha': self.alpha,
            'alpha_genius': self.alpha_genius,
            'sigma': self.sigma,
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
            filename = f"{self.name}_v{self.version}.json"
        file = path / filename
        if file.is_file():
            with open(filename, 'r') as f:
                state = json.load(f)
            self.kiyoo = np.array(state['Q'])
            self.alpha = state['alpha']
            self.alpha_genius = state['alpha_genius']
            self.sigma = state['sigma']
            self.bias = state['bias']
            self.choices = state['choices']
            self.rewards = state['rewards']
            self.history = [np.array(q) for q in state['history']]
        else:
            print(f"File {filename} not found. Agent state not loaded.")

    @property
    def params(self):
        return self.alpha, self.decay, self.sigma, self.bias

    @params.setter
    def params(self, values):
        self.alpha, self.decay, self.sigma, self.bias = values


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
        self.bounds = kwargs.get("bounds", [
            (1e-5, 1),  # alpha
            (1e-5, 1),  # sigma
            (-np.inf, np.inf)  # bias
            ])

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        self.kiyoo[action] = self.kiyoo[action] + self.alpha * (reward - self.kiyoo[action])

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
        self.bounds = kwargs.get("bounds", [
            (1e-5, 1),  # alpha
            (1e-5, 1),  # alpha_genius
            (1e-5, 1),  # sigma
            (-np.inf, np.inf)  # bias
            ])

    def update(self, action, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.

        Args:
            action (int): action chosen by the agent
            reward (float): reward received from the environment
        """
        if reward == 1:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha * (1 - self.kiyoo[action])
        else:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha_genius * (0 - self.kiyoo[action])

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

        self.sigma = sigma
        self.bias = bias
        self.omega = omega
        self.bounds = kwargs.get("bounds", [
            (1e-5, 1),  # alpha
            (1e-5, 1),  # sigma
            (-np.inf, np.inf),  # bias
            (1e-5, 1)  # omega
            ])

        # Initialize the other set of values
        self.vee = np.zeros(self.kiyoo.shape)
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
        self.kiyoo[action] = self.kiyoo[action] + self.alpha * (1 - self.kiyoo[action])

        # Update the feature values
        self.vee[feature] = self.vee[feature] + self.alpha * (1 - self.vee[feature])

        # Save history
        self.choices.append(action)
        self.rewards.append(reward)
        self.history.append(self.kiyoo.copy())
        self.vee_history.append(self.vee.copy())

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
        self.params = params
        self.reset()
        log_like = 0
        for trial in data:

            # Unpack the trial
            action, reward, feat_left = trial

            # Get the feature choice
            feature = feat_left if action == 0 else 1 - feat_left  # left and right features are complementary

            # Update the Q-values
            self.update(action, feature, reward)

            # Compute the choice probabilities
            probs = self.get_choice_probs(feat_left)

            # Compute the log-likelihood
            ll = np.log(probs[action] + 1e-10)
            self.losses.append(ll)
            log_like += ll

        return -log_like
