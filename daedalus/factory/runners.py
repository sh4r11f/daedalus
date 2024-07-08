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
import numpy as np


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
        """
        Shuffle all blocks.
        """
        # Shuffle
        random.shuffle(self.blocks)

        # Ensure the practice block is first
        self.put_practice_first()

        # Reset the block ID and index
        for idx, block in enumerate(self.blocks):
            block.idx = idx
            block.id = idx + 1

    def put_practice_first(self):
        """
        Move the practice block to the first position in the list of blocks.
        """
        index = [idx for idx, block in enumerate(self.blocks) if block.name == "practice"][0]
        practice_block = self.blocks[index]
        self.blocks = [practice_block] + self.blocks[:index] + self.blocks[index+1:]

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
                conc_blocks.append(block)
        return conc_blocks

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.blocks):
            result = self.blocks[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def next(self):
        return self.__next__()


class BlockFactory:
    def __init__(self, block_idx, block_info):

        self.idx = block_idx
        self.id = block_idx + 1
        self.info = block_info
        self.name = block_info["name"]

        self.trials = []

        self.duration = 0
        self.error = None
        self.repeat = False
        self.needs_calib = False

        self.failed = 0
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

    def add_trials(self, trials):
        for trial in trials:
            self.add_trial(trial)

    def get_trial(self, trial_idx):
        return self.trials[trial_idx]

    def shuffle(self):
        """
        Shuffle all trials.
        """
        # Shuffle
        random.shuffle(self.trials)

        # Reset the trial ID and index
        for idx, trial in enumerate(self.trials):
            trial.idx = idx
            trial.id = idx + 1

    def recycle(self):
        """
        Determine if the block will repeat.
        """
        self.failed += 1

        for trial in self.trials:
            trial.reset()

        self.needs_calib = True

        self.duration = 0
        self.error = None
        self.repeat = False

        self.failed = 0
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.trials):
            result = self.trials[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def next(self):
        return self.__next__()


class TrialFactory:
    def __init__(self, trial_idx, **kwargs):

        # Setup
        self.idx = trial_idx
        self.id = trial_idx + 1
        self.name = kwargs.get("name", None)

        self._frames = []
        self.fix_frames = []
        self.view_frames = []
        self.cue_frames = []
        self.resp_frames = []
        self.feedback_frames = []
        self.mask_frames = []
        self.break_frame = -1

        self.fix_times = []
        self.view_times = []
        self.cue_times = []
        self.resp_times = []
        self.feedback_times = []
        self.mask_times = []

        self.eye_events = []
        self.eye_samples = []
        self.frame_intervals = []

        self.duration = 0
        self.drift_correction = 0
        self.repeat = False
        self.key_press = None
        self.response = None
        self.saccade = None
        self.error = None

        self.failed = 0
        self.index = 0

        # Set and overwrite the attributes that are provided
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def n_frames(self):
        return len(self._frames)

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, frames):
        self._frames = frames

    def recycle(self):
        """
        Reset for a repeated trial.
        """
        self.failed += 1
        self.index = 0

        self.fix_times = np.zeros(len(self.fix_frames))
        self.view_times = np.zeros(len(self.view_times))
        self.cue_times = np.zeros(len(self.cue_times))
        self.resp_times = np.zeros(len(self.resp_times))
        self.feedback_times = np.zeros(len(self.feedback_times))
        self.mask_times = np.zeros(len(self.mask_times))

        self.eye_events = []
        self.eye_samples = []
        self.frame_intervals = []

        self.duration = 0
        self.drift_correction = 0
        self.repeat = False
        self.key_press = None
        self.response = None
        self.saccade = None
        self.error = None

    def reset(self):
        """
        Reset for a block.
        """
        self.recycle()
        self.failed = 0

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

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self._frames):
            result = self._frames[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def next(self):
        return self.__next__()
