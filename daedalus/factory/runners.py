#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                    SCRIPT: trials.py
#
#
#               DESCRIPTION: Class to hold trial information
#
#
#                      RULE: DAYW
#
#
#
#                   CREATOR: Sharif Saleki
#                      TIME: 06-30-2024-7810598105114117
#                     SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
import random


class TaskFactory:
    def __init__(self, task_name, params):

        # Setup
        self.name = task_name
        self.params = params

        # Blocks
        self.info = None
        self.blocks = []
        self.index = 0

    @property
    def n_blocks(self):
        return len(self.blocks)

    def load_blocks(self):
        self.info = self.concat_blocks()

    def shuffle(self):
        random.shuffle(self.blocks)
        self.put_practice_first()

    def shuffle_except_practice(self):

        index = [idx for idx, block in enumerate(self.blocks) if block.name == "practice"]

        # Extract the element to keep constant
        constant_element = self.blocks[index]

        # Create a new array excluding the constant element
        arr_to_shuffle = self.blocks[:index] + self.blocks[index+1:]

        # Shuffle the new array
        random.shuffle(arr_to_shuffle)

        # Insert the constant element back into its original position
        self.blocks = arr_to_shuffle[:index] + [constant_element] + arr_to_shuffle[index:]

    def put_practice_first(self):
        index = [idx for idx, block in enumerate(self.blocks) if block.name == "practice"]
        practice_block = self.blocks[index]
        self.blocks = [practice_block] + self.blocks[:index] + self.blocks[index+1:]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.blocks):
            result = self.blocks[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def add_block(self, block, idx=None):
        if idx is not None:
            self.blocks.insert(idx, block)
        else:
            self.blocks.append(block)

    def get_block(self, block_idx):
        return self.blocks[block_idx]

    def concat_blocks(self):
        conc_blocks = []
        for block in self.params["blocks"]:
            for _ in range(block["n_repeats"]):
                block.pop("n_repeats", None)
                conc_blocks.append(block)
        return conc_blocks


class BlockFactory:
    def __init__(self, block_idx, block_info):

        self.idx = block_idx
        self.id = block_idx + 1
        self.info = block_info
        self.name = block_info["name"]
        self.repeat = None
        self.repeated = False
        self.needs_calib = False
        self.trials = []
        self.n_trials = 0
        self.index = 0
        self._handler = None

    @property
    def n_trials(self):
        return len(self.trials)

    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, handler):
        self._handler = handler

    def add_trial(self, trial, idx=None):
        if idx is not None:
            self.trials.insert(idx, trial)
        else:
            self.trials.append(trial)

    def get_trial(self, trial_idx):
        return self.trials[trial_idx]

    def shuffle(self):
        random.shuffle(self.trials)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.trials):
            result = self.trials[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration


class TrialFactory:
    def __init__(self, trial_idx, **kwargs):

        # Setup
        self.idx = trial_idx
        self.id = trial_idx + 1
        self.duration = None
        self.frames = []
        self.eye_events = []
        self.eye_samples = []
        self.key_press = None
        self.frame_intervals = None
        self.drift_correction = 0
        self.repeat = False
        self.repeated = False
        self.fake_response = None
        self.fake_response_onset = -1

        # Set and overwrite the attributes that are provided
        for key, value in kwargs.items():
            setattr(self, key, value)

    def s2ms(self, var_name):
        # Get the value
        value = getattr(self, var_name)
        # If a list convert all elements
        if isinstance(value, list):
            setattr(self, var_name, [int(val * 1000) for val in value])
        else:
            setattr(self, var_name, int(value * 1000))

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def to_dict(self):
        return self.__dict__
