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


class TaskFactory:
    def __init__(self, exp_config, task_id):
        self.exp_params = exp_config
        self.task_params = self.exp_params[task_id]
        self.task_id = task_id

        self.blocks = BlockFactory(self.exp_params, self.task_id)
        self.n_blocks = self.blocks.n_total

    def create_task(self):
        pass


class TrialFactory:
    def __init__(self, trial_id, conditions):
        self.id = trial_id
        self.cond = conditions


class BlockFactory:
    def __init__(self, exp_config, task_id):

        self.exp_params = exp_config
        self.task_params = self.exp_params[task_id]
        self.task_id = task_id

        self.all = self.concat_blocks()
        self.n_total = len(self.all)

    def concat_blocks(self):
        conc_blocks = []
        for block in self.settings.exp["Tasks"][self.task_id]["blocks"]:
            for _ in range(block["n_blocks"]):
                conc_blocks.append(block)
        return conc_blocks

    def create_block(self):
        pass