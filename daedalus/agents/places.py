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
import numpy as np
import pandas as pd


class BaseEnv:
    def __init__(self, name, root, **kwargs):
        self.name = name
        self.root = root
        self._data = kwargs.get("data", None)
        if self._data is not None:
            self.rewards = self._data["Reward"].values
            self.n_trials = len(self._data)
        else:
            self.rewards = np.array([])
            self.n_trials = 0

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.rewards = self._data["Reward"].values

    def init_dataset(self, name, data):
        df = pd.DataFrame(data)
        df["Name"] = name
        setattr(self, name, df)

        return df

    def record_data(self, event_data, data_name=None):
        edf = pd.DataFrame(event_data)
        if data_name is None:
            self.data = pd.concat([self.data, edf], ignore_index=True)
        else:
            df = getattr(self, data_name)
            df = pd.concat([df, edf], ignore_index=True)
            setattr(self, data_name, df)

    def record_event(self, data_name=None, **kwargs):

        event = {key: [value] for key, value in kwargs.items()}
        edf = pd.DataFrame(event)
        if data_name is None:
            self.data = pd.concat([self.data, edf], ignore_index=True)
        else:
            df = getattr(self, data_name)
            df = pd.concat([df, edf], ignore_index=True)
            setattr(self, data_name, df)

    def mask_event(self, event_column, values):
        mask = self.data[event_column].isin(values)
        return mask, mask.sum()

    def save_data(self, data_name=None, filename=None):

        path = self.root / "data" / "models"
        path.mkdir(parents=True, exist_ok=True)

        if data_name is None:
            data = getattr(self, self.name)
        else:
            data = self._data

        if filename is None:
            filename = f"{self.name}_{data['Name']}.csv"

        file = path / filename
        data.to_csv(file, index=False)


class Colosseum(BaseEnv):
    def __init__(self, name, root, **kwargs):
        super().__init__(name, root, **kwargs)

        self.warriors = []
        self.animals = []
        self.followers = []
        self.audience = []

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

    def add_warrior(self, warrior):
        self.warriors.append(warrior)

    def add_follower(self, name, choice_column, action_map, state_map=None):
        choices = self.data[choice_column].values
        flr = Follower(name, choices, action_map, state_map)
        flr.rewards = self.rewards

        # Add the animal to the colosseum
        setattr(self, name, flr)
        self.followers.append(flr)

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


class Follower:
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
