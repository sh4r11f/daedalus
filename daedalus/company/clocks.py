#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: clocks.py
#
#
#                   DESCRIPTION: Time keeprs
#
#
#                          RULE: DAYW
#
#
#
#                       CREATOR: Sharif Saleki
#                          TIME: 07-20-2024-7810598105114117
#                         SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from datetime import date
import time


class TimeKeeper:
    def __init__(self, **kwargs):

        self.name = kwargs.get("name")

        self.today = date.today().strftime("%b-%d-%Y")
        self.init = time.time()

        self._tick = -1
        self._tock = -1
        self.ttdur = None
        self.t0 = -1
        self.tn = -1

    def time(self):
        return time.time()

    def tick(self, duration):
        self._tick = time.time()

    def tock(self):
        self._tock = time.time()
        self.ttdur = self._tock - self._tick
        self._tock = -1
        return self.ttdur
