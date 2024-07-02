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
from datetime import date

class TimeManager:
    def __init__(self):

        self.exp = core.Clock()
        self.block = core.Clock()
        self.trial = core.MonotonicClock()
        self.today = date.today().strftime("%b-%d-%Y")
        self._tick = -1
        self._ttdur = -1
        self._tock = False

        self.exp_events = {}
        self.block_events = {}
        self.trial_events = {}
        self.tracker_times = []
        self.frame_intervals = []
        self.tracker_times = []

    def tick(self, duration):
        self._tock = False
        self._ttdur = duration
        self._tick = self.exp.getTime()

    @property
    def tock(self):
        self._tock = self.duration_over(self._tick, self._ttdur)
        return self._tock

    def duration_over(self, t0, duration):
        return self.exp.getTime() - t0 > duration

    def start(self):
        self.exp.reset()
        self.block.reset()
        self.trial = core.MonotonicClock()

    def start_block(self):
        self.block.reset()

    def start_trial(self):
        self.trial = core.MonotonicClock()

    def record_exp(self, event_name):
        self.exp_events[event_name] = self.exp.getTime()

    def record_block(self, event_name):
        self.block_events[event_name] = self.block.getTime()

    def record_trial(self, event_name):
        self.trial_events[event_name] = self.trial.getTime()

    def set_stim_durations(self, durations):
        """
        Sets duration attributes,
        """
        for key, value in durations.items():
            setattr(self, key, value)
