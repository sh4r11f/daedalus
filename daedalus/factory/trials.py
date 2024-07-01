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


class Trial:
    def __init__(self, trial_id, trial_type, clock):
        self.id = trial_id
        self.type = trial_type
        self.clock = clock

        self.start = 

    def __str__(self):
        return f"Trial {self.trial_id}: {self.trial_type}"