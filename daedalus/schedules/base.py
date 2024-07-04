#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================== #
#
#
#                     SCRIPT: reward.py
#
#
#                DESCRIPTION: Reward schedules for the task
#
#
#                       RULE: DAYW
#
#
#
#                  CREATOR: Sharif Saleki
#                         TIME: 07-04-2024-7810598105114117
#                       SPACE: Dartmouth College, Hanover, NH
#
# ======================================================================================== #
from pathlib import Path

import numpy as np

from daedalus import utils


class RewardSchedule:
    """
    Base class for all reward schedules

    Args:
        version (str): Version of the rMIB experiment
        root (str): Root directory for the rMIB experiment
        subject (str): Subject ID
        session (str): Session ID
        task (str): Task name
    """
    def __init__(self, root, params):
        """
        Initialize the base reward schedule
        """
        # Set attributes
        self.root = Path(root)
        self.params = params

        # Schedule variables
        self.choices = []
        self.choice_probs = [0.5, 0.5]
        self.reward_probs = [0.5, 0.5]

    def add_choice(self, choice):
        """
        Add a choice to the reward schedule

        Args:
            choice (str): The choice to add
        """
        self.choices.append(choice)

    def update(self, choice):
        """
        Update the reward schedule based on the choice

        Args:
            choice (str): The choice to update
        """
        self.choices.append(choice)
        p_left = 1 - np.mean(self.choices[-self.params["window_size"]:])
        self.choice_probs = [p_left, 1 - p_left]
        self.reward_probs = self.get_reward_probs(p_left)
