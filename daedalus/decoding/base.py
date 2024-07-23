#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: base.py
#
#
#                   DESCRIPTION: Base decoder class
#
#
#                          RULE: DAYW
#
#
#
#                       CREATOR: Sharif Saleki
#                          TIME: 07-18-2024-7810598105114117
#                         SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from pathlib import Path
from joblib import Memory
# from shutil import rmtree

from daedalus.company.clocks import TimeKeeper


class Decoder:
    def __init__(self, name, cache_dir, params, **kwargs):

        self.name = name
        self.cache_dir = cache_dir
        self.params = params

        self.clf = None

        self.memory = Memory(location=cache_dir, verbose=self.params["verbose"])

    def clean(self):
        self.memory.clear(warn=False)


class ClusterDuck:
    def __init__(self, name, cache_dir, params, **kwargs):

        self.name = name
        self.caceh_dir = cache_dir
        self.params = params

        self.memory = Memory(location=cache_dir, verbose=self.params["verbose"])
