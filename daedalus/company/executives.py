#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: executives.py
#
#
#               DESCRIPTION: High-level project managers
#
#
#                           RULE: DAYW
#
#
#
#                      CREATOR: Sharif Saleki
#                            TIME: 07-16-2024-7810598105114117
#                          SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from daedalus import utils


class CEO:
    def __init__(self, name, root, version, **kwargs):

        self.name = name
        self.root = root
        self.version = version
        self.settings = utils.read_config(self.root / "config" / "settings.yaml")
        