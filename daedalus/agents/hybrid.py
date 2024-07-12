#!/usr/bin/env python
"""
created 3/6/2024

@author Sharif Saleki

Description:
"""
import numpy as np

from .base import BaseRL
from .simple import SimpleQ


class FeatureAction(SimpleQ):
    """
    """
    def __init__(
        self,
        name, root, version,
        alpha_act=0.5, alpha_feat=0.5, decay_act=0.9, decay_feat=0.9, beta=1, bias=0, omega=0.5,
        **kwargs
        ):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, root, version, **kwargs)

        self._V = np.zeros(self.n_actions).reshape(-1, 1)
        self._Q = np.zeros(self.n_actions)
        self._C = np.zeros((self.n_actions, self.n_actions))  # action x feature

        self.action_choices = []
        self.feature_choices = []
        self.action_history = []
        self.feature_history = []
        self.combined_history = []

        self._n_params = 6
        self._alpha_act = alpha_act
        self._alpha_feat = alpha_feat
        self._decay_act = decay_act
        self._decay_feat = decay_feat
        self._beta = beta
        self._omega = omega
        bounds = [
            (1e-5, 1),  # alpha_act
            (1e-5, 1),  # alpha_feat
            (1e-5, 1),  # decay_act
            (1e-5, 1),  # decay_feat
            (1, 10),  # beta
            (0, 1),  # omega
        ]
        self.bounds = kwargs.get("bounds", bounds)

        self.bias = bias

    def reset(self):
        """
        Reset the agent.
        """
        self._V = np.zeros(self.n_actions).reshape(-1, 1)
        self._Q = np.zeros(self.n_actions)
        self._C = np.zeros((self.n_actions, self.n_actions))
        self.action_choices = []
        self.feature_choices = []
        self.action_history = []
        self.feature_history = []
        self.combined_history = []
        self.rewards = []

    def update(self, action_choice, feature_choice, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.
        """
        # Update action
        self._V[action_choice] += self._alpha_act * (reward - self._V[action_choice])
        self._V[1 - action_choice] *= (1 - self._decay_act)

        # Update feature
        self._Q[feature_choice] += self._alpha_feat * (reward - self._Q[feature_choice])
        self._Q[1 - feature_choice] *= (1 - self._decay_feat)

        # Update combined
        self._C = self._omega * self._V + (1 - self._omega) * self._Q

        # Save history
        self.action_choices.append(action_choice)
        self.feature_choices.append(feature_choice)
        self.rewards.append(reward)
        self.feature_history.append(self._Q.copy())
        self.action_history.append(self._V.copy())
        self.combined_history.append(self._C.copy())

    def choose_action(self, feature_left):
        """
        Chooses an action based on the current state.

        Returns:
            int: action chosen by the agent
        """
        probs = self.get_choice_probs(feature_left)
        action_choice = 0 if np.random.rand() < probs[0] else 1
        feature_choice = feature_left if action_choice == 0 else 1 - feature_left
        return action_choice, feature_choice

    def get_choice_probs(self, feature_left):
        """
        Compute the choice probabilities.
        """
        diff = self._C[0, feature_left] - self._C[1, 1 - feature_left]
        logits = self._beta * diff + self.bias
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
            action_choice, feature_choice, feature_left, reward = trial
            self.update(action_choice, feature_choice, reward)
            probs = self.get_choice_probs(feature_left)
            nll -= np.log(probs[action_choice] + 1e-10)
        return nll

    def set_params(self, params):
        """
        Update model parameters.
        """
        (
            self._alpha_act,
            self._alpha_feat,
            self._decay_act,
            self._decay_feat,
            self._beta,
            self._omega
        ) = params

    @property
    def params(self):
        return [self._alpha_act, self._alpha_feat, self._decay_act, self._decay_feat, self._beta, self._omega]

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        self._V = value

    @property
    def alpha_act(self):
        return self._alpha_act

    @alpha_act.setter
    def alpha_act(self, value):
        self._alpha_act = value

    @property
    def alpha_feat(self):
        return self._alpha_feat

    @alpha_feat.setter
    def alpha_feat(self, value):
        self._alpha_feat = value

    @property
    def decay_act(self):
        return self._decay_act

    @decay_act.setter
    def decay_act(self, value):
        self._decay_act = value

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value


class ObjectBased(BaseRL):

    def __init__(
        self,
        name, root, version,
        alpha=0.5, decay=0.9, beta=1, bias=0,
        **kwargs
    ):
        """
        Initialize the Q-learning agent
        """
        super().__init__(name, root, version, n_actions=4, n_states=1, **kwargs)

        self._Q = np.zeros(self.n_actions)
        self._alphas = np.repeat([alpha], 4)
        self._decays = np.repeat([decay], 4)
        self._beta = beta

        self._n_params = 9
        bounds = [
            (1e-5, 1),  # alpha
            (1e-5, 1),  # alpha
            (1e-5, 1),  # alpha
            (1e-5, 1),  # alpha
            (1e-5, 1),  # decay
            (1e-5, 1),  # decay
            (1e-5, 1),  # decay
            (1e-5, 1),  # decay
            (1, 100),  # beta
            # (0, 1),  # bias
        ]
        self.bounds = kwargs.get("bounds", bounds)

        self._bias = bias

    def reset(self):
        """
        Reset the agent.
        """
        self._Q = np.zeros(self.n_actions)
        self.choices = []
        self.rewards = []
        self.history = []

    def update(self, choice, reward):
        """
        Update the Q-value for the given state-action pair based on the reward received and the maximum Q-value for the
        next state.
        """
        # Update action
        for i in range(self.n_actions):
            if i == choice:
                self._Q[i] += self._alphas[i] * (reward - self._Q[i])
            else:
                self._Q[i] *= (1 - self._decays[i])

        # Save history
        self.choices.append(choice)
        self.rewards.append(reward)
        self.history.append(self._Q.copy())

    def choose_action(self, options):
        """
        Chooses an action based on the current state.

        Returns:
            int: action chosen by the agent
        """
        probs = self.get_choice_probs(options)

        return options[0] if np.random.rand() < probs[0] else options[1]

    def get_choice_probs(self, options):
        """
        Compute the choice probabilities.
        """
        diff = self._Q[options[0]] - self._Q[options[1]]
        logits = self._beta * diff + self._bias
        prob_a = self.sigmoid(logits)
        return [prob_a, 1 - prob_a]

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
            choice, reward = trial
            options = [choice, 3 - choice] if choice < 2 else [3 - choice, choice]
            self.update(choice, reward)
            probs = self.get_choice_probs(options)
            nll -= np.log(probs[choice] + 1e-10)
        return nll

    def set_params(self, params):
        """
        Update model parameters.
        """
        (
            self._alphas[0],
            self._alphas[1],
            self._alphas[2],
            self._alphas[3],
            self._decays[0],
            self._decays[1],
            self._decays[2],
            self._decays[3],
            self._beta,
        ) = params

    @property
    def params(self):
        return [
            self._alphas[0],
            self._alphas[1],
            self._alphas[2],
            self._alphas[3],
            self._decays[0],
            self._decays[1],
            self._decays[2],
            self._decays[3],
            self._beta,
        ]

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value

    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, value):
        self._alphas = value

    @property
    def decays(self):
        return self._decays

    @decays.setter
    def decays(self, value):
        self._decays = value

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
