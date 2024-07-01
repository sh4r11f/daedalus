#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                    SCRIPT: timing.py
#
#
#               DESCRIPTION: Timing manager
#
#
#                      RULE: DAYW
#
#
#
#                   CREATOR: Sharif Saleki
#                      TIME: 07-01-2024-7810598105114117
#                     SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from psychopy import core


class TimeManager:
    def __init__(self):

        self.exp = core.Clock()
        self.block = core.Clock()
        self.trial = core.MonotonicClock()

        self.exp_end_time = None
        self.block_end_time = None
        self.trial_end_time = None

    def start(self):
        self.exp.reset()
        self.block.reset()
        self.trial.reset()

    def start_block(self):
        self.block.reset()

    def start_trial(self):
        self.trial = core.MonotonicClock()

    def exp_duration(self):
        return self.exp.getTime()

    def block_duration(self):
        return self.block.getTime()

    def trial_duration(self):
        return self.trial.getTime()
