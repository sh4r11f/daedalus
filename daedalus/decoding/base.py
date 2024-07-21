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
from shutil import rmtree

from daedalus.company.clocks import TimeKeeper
from daedalus.company.managers import DataManager, FileManager
from daedalus.log_handler import DaedalusLogger


class Decoder:
    def __init__(self, name, params, data, labels, log_dir, cache_dir, **kwargs):

        self.name = name
        self.params = params

        self.clf = None

        self.data = DataManager()
        self.data.add(raw=data, labels=labels)

        self.files = FileManager()
        self.files.add(log=Path(log_dir), cache=Path(cache_dir))

        self.clock = TimeKeeper(name="Decoder")
        self.logger = DaedalusLogger("Decoder", log_file=self.files.log)

        self.memory = Memory(location=self.files.cache, verbose=self.params["verbose"])

    def turn_on(self):
        self.logger.info("Turning on the decoder...")

    def turn_off(self):
        self.logger.info("Turning off the decoder...")
        self.memory.clear(warn=False)
        rmtree(self.files.cache)
        self.logger.info("Goodbye!")


class ClusterDuck:
    def __init__(self, name, data, params, cache_dir, **kwargs):

        self.name = name
        self.model = None

        self.data = DataManager()
        self.data.add(raw=data)

        self.files = FileManager()
        self.files.add(cache=cache_dir)

        self.logger = DaedalusLogger("Clusterer", log_file=self.files.log)

        self.memory = Memory(location=self.files.cache, verbose=self.params["verbose"])
