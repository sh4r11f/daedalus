#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: clowns.py
#
#
#               DESCRIPTION: Object-based RLs
#
#
#                           RULE: 
#
#
#
#                      CREATOR: Sharif Saleki
#                            TIME: 07-12-2024-7810598105114117
#                          SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
import random
import numpy as np

from .base import BaseGent


class ObjectBased(BaseGent):
    def __init__(self, name, sigma=0.5, bias=0.5, **kwargs):
        super().__init__(name, n_actions=4, n_states=1, **kwargs)

        self.sigma = sigma
        self.bias = bias
        self._params.append(["sigma", self.sigma])
        self._params.append(["bias", self.bias])

        self.kiyoo = np.zeros(self.n_actions)
        self.bounds.append(kwargs.get("sigma_bounds", ("sigma", (1e-5, 1))))
        self.bounds.append(kwargs.get("bias_bounds", ("bias", (-np.inf, np.inf))))

    def update(self, action, reward):
        self.kiyoo[action] = self.kiyoo[action] + self.alpha * (reward - self.kiyoo[action])

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
            action, reward, left_feat = trial

            # Find available options
            feature = left_feat if action == 0 else 1 - left_feat

            # Find object number
            chosen = action * 2 + feature
            unchosen = (1 - action) * 2 + (1 - feature)
            options = [chosen, unchosen]

            # Update the value
            self.update(action, reward)

            # Calculate the loss
            probs = self.get_choice_probs(options)
            log_like = np.log(probs[action])
            self.hoods.append(log_like)

            nll -= log_like

        return nll


class ObjectRewUnrew(ObjectBased):
    def __init__(self, name, alpha_unr=0.5, **kwargs):
        super().__init__(name, **kwargs)

        self.alpha_unr = alpha_unr
        self._params.append(["alpha_unr", self.alpha_unr])
        self.bounds.append(kwargs.get("alpha_unr_bounds", ("alpha_unr", (1e-5, 1))))

    def update(self, action, reward):
        # Update rewarded option
        if reward == 1:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha * (reward - self.kiyoo[action])
        # Update unrewarded options
        else:
            self.kiyoo[action] = self.kiyoo[action] + self.alpha_unr * (reward - self.kiyoo[action])


class ObjectRewUnrewDecay(ObjectRewUnrew):
    def __init__(self, name, decay=0.5, **kwargs):

        super().__init__(name, **kwargs)
        self.decay = decay
        self._params.append(["decay", self.decay])
        self.bounds.append(kwargs.get("decay_bounds", ("decay", (1e-5, 1))))

    def update(self, action, reward):
        # Update chosen option
        super().update(action, reward)

        # Decay everything else
        others = np.where(np.arange(4) != action)[0]
        for q in others:
            self.kiyoo[q] = self.kiyoo[q] - self.decay * self.kiyoo[q]
