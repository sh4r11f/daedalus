#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                    SCRIPT: environments.py
#
#
#               DESCRIPTION: Environment classes for reinforcement learning agents.
#
#
#                      RULE: DAYW
#
#
#
#                   CREATOR: Sharif Saleki
#                      TIME: 07-10-2024-7810598105114117
#                     SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
import pandas as pd
import numpy as np


class Colosseum:
    def __init__(self, name, root, data=None):
        self.name = name
        self.root = root
        self.warriors = []
        self.animals = []
        self.audience = []
        if data is None:
            self.data = pd.DataFrame()
            self.rewards = np.array([])
        else:
            self.data = data
            self.rewards = data.get(["Reward"], pd.Series([])).values

        self.n_trials = self.data.shape[0]

    def fuse(self, choices, rewards):

        train = list(zip(choices, rewards))
        test = []
        for r in range(len(choices)):
            options = np.zeros(2)
            options[choices[r]] = rewards[r]
            options[1 - choices[r]] = 1 - rewards[r]
            test.append(options)

        return train, test

    def sample(self):
        pass

    def event_mask(self, event_column, values):
        mask = self.data[event_column].isin(values)
        return mask, mask.sum()

    def add_warrior(self, warrior):
        self.warriors.append(warrior)

    def add_follower(self, name, choice_column, action_map, state_map=None):
        choices = self.data[choice_column].values
        animal = Animal(name, choices, action_map, state_map)
        animal.rewards = self.rewards

        # Add the animal to the colosseum
        setattr(self, name, animal)
        self.animals.append(animal)

    def add_audience(self, audience):
        self.audience.append(audience)

    def get_training_data(self, choice_col, reward_col, n_samples=None, replace=False):

        if n_samples is None:
            n_samples = len(self.data)

        indices = np.random.choice(len(self.data), n_samples, replace=replace)
        choices = self.data.loc[indices, choice_col].values
        rewards = self.data.loc[indices, reward_col].values

        trials = list(zip(choices, rewards))
        return trials

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data


class Animal:
    def __init__(self, name, choice_history, action_map, state_map=None):
        self.name = name
        self.choices = choice_history
        self.action_map = action_map
        self.state_map = [(0, "existence")] if state_map is None else state_map

        self.rewards = []
        self.possible_actions = [action for _, action in self.action_map]
        self.possible_states = [state for _, state in self.state_map]
        self.choice_probs = np.zeros((len(self.possible_states), len(self.possible_actions)))

    def state(self):
        pass

    def add_choice(self, choice):
        self.choices.append(choice)

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
